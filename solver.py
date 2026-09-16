import torch
import torch.nn as nn
import numpy as np
import os
from model.AdaTrendLike import AdaTrendLike
from data_factory.data_loader import get_loader_segment
from tqdm import tqdm
from utils.evaluation import get_bestF1
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score,accuracy_score
import time
import datetime as dt

class TrendLike(nn.Module):
    def __init__(self):
        super(TrendLike, self).__init__()
        
    def forward(self, rec_d,true_x,mu_trend,true_trend,scale_tril,pi_hat,y_label):
        B, L,Fea = true_x.shape
        pi_hat = pi_hat.detach().cpu().reshape(-1)  
        mu_trend=mu_trend.detach().cpu().reshape(-1,Fea)
        scale_tril=scale_tril.detach().cpu().reshape(-1,Fea,Fea)
        anomalyscore= torch.full_like(pi_hat, float('0'))  
        rec_d=rec_d.detach().cpu().reshape(-1,Fea)
        true_x=true_x.detach().cpu().reshape(-1,Fea)
        true_trend=true_trend.detach().cpu().reshape(-1,Fea)
        y_label=y_label.detach().cpu().reshape(-1)  
        windowroll=L
        for l in range(0,L*B):
            l_min = max(l - windowroll, 0)   
            l_max = min(l + windowroll, L*B)
            pi_hat_bar = torch.log_softmax(pi_hat[l_min:l_max], -1) 
            mnpos=torch.distributions.multivariate_normal.MultivariateNormal(mu_trend[l_min:l_max,:],scale_tril=scale_tril[l_min:l_max,:,:])
            likelihoodpos =mnpos.log_prob(true_x[l_min:l_max,:]-rec_d[l_min:l_max,:])
            anomalyscore[l] = torch.logsumexp(pi_hat_bar+likelihoodpos, dim=-1).unsqueeze(-1)
        return -anomalyscore,pi_hat


class Solver(object):
    DEFAULTS = {}
    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)
        _,dimret = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size, step=self.stride,
                                                mode='train',
                                                dataset=self.dataset,val_ratio=self.val_ratio,noise_ratio=self.noise_ratio)

        self.dimret=dimret
        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pterrterion =  nn.MSELoss()
        self.TrendLike = TrendLike()
        keys_for_filename = [
            k for k, v in config.items()
            if isinstance(v, (int, float, str)) and k not in ["model_save_path", "data_path", "mode"]
        ]
        filename = "_".join(f"{k}{config[k]}" for k in keys_for_filename) + "_best.pth"
        self.filepath = f"{config['model_save_path']}/{filename}"
        print(self.filepath)
    def build_model(self):
        model = AdaTrendLike(win_size=self.win_size ,c_out=self.dimret, d_model=self.d_model,dlayer=self.dlayer,movk=self.movk,freqk=self.freqk,flayer=self.flayer,cnngroup=self.cnngroup) 
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {num_params / 1e6:.2f}M")
        self.model = nn.DataParallel(model, device_ids=['cuda:0'])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )

    def loss_function(self, rec_d, true_x, mu_trend, true_trend, scale_tril, pi_hat):
        rec_loss = self.pterrterion(rec_d, true_x - true_trend)
        trend_dist = torch.distributions.MultivariateNormal(
            mu_trend, scale_tril=scale_tril
        )
        pi_hat_bar = torch.log_softmax(pi_hat, -1) 
        LL_loss = torch.logsumexp(pi_hat_bar+trend_dist.log_prob(true_trend),-1)
        loss = rec_loss.mean() -self.lamda * LL_loss.mean()
        return loss

    def train(self):
        print("======================TRAIN MODE======================")
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
    
        best_val = float("inf")
        patience = 2 

        for epoch in range(self.num_epochs):
            self.train_loader,_ = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size, step=self.stride,
                                                mode='train',
                                                dataset=self.dataset,val_ratio=self.val_ratio,noise_ratio=self.noise_ratio)
            self.val_loader,_ = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size, step=self.stride,
                                                    mode='val',
                                                    dataset=self.dataset,val_ratio=self.val_ratio,noise_ratio=self.noise_ratio)

            self.model.train()
            epoch_loss = []
            start_time = time.time()
            for i, (true_x, _) in enumerate(tqdm(self.train_loader)):
                true_x = true_x.float().to(self.device)
                self.optimizer.zero_grad()
                rec_d, true_trend, mu_trend, scale_tril, pi_hat = self.model(true_x)
                loss = self.loss_function(rec_d, true_x, mu_trend, true_trend, scale_tril, pi_hat)
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.item())
            train_loss = sum(epoch_loss) / len(epoch_loss)
            torch.cuda.synchronize()  # Make sure all GPU ops are done
            end_time = time.time()
            print(f"Epoch time: {end_time - start_time:.3f} seconds")

            # =======================
            #       VALIDATION
            # =======================
            self.model.eval()
            val_LL_list = []

            with torch.no_grad():
                for i, (true_x, _) in enumerate(tqdm(self.val_loader)):
                    true_x = true_x.float().to(self.device)
                    rec_d, true_trend, mu_trend, scale_tril, pi_hat = self.model(true_x)
                    loss = self.loss_function(rec_d, true_x, mu_trend, true_trend, scale_tril, pi_hat)
                    val_LL_list.append(loss.detach().reshape(-1))
            val_scores = torch.cat(val_LL_list, dim=0)
            val_mean = val_scores.mean().item()
            print(
                f"Epoch [{epoch+1}/{self.num_epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val_LL: {val_mean:.4f} | "
            )
            if val_mean< best_val:
                best_val = val_mean
                wait = 0
                print("Best Saved")
                torch.save(
                    self.model.state_dict(),
                    self.filepath
                    )
            else:
                wait += 1
                print(f"Early stopping wait at {wait}")
                if wait >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        return val_mean

    def test(self):
        self.model.load_state_dict(torch.load(self.filepath))
        self.model.eval()
        print("======================TEST MODE======================")
        A_score_list=[]
        test_y_label=[]
        rec_d_list=[]
        mu_trend_list=[]
        scale_tril_list=[]
        true_x_list=[]
        pi_hat_list=[]
        self.test_loader,_ = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size, step=self.win_size,
                                    mode='test',
                                    dataset=self.dataset,val_ratio=self.val_ratio,noise_ratio=self.noise_ratio)
        with torch.no_grad():
            total_inference_time=0
            for i, (true_x, y_label) in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):

                B, L, Fea = true_x.shape
                true_x = true_x.float().to(self.device)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start = time.time()

                rec_d, true_trend, mu_trend, scale_tril, pi_hat = self.model(true_x)
                true_trend = true_trend.to(self.device)
                trendLL,pi_hat = self.TrendLike(
                    rec_d, true_x, mu_trend, true_trend, scale_tril, pi_hat, y_label
                )
                if torch.cuda.is_available():
                        torch.cuda.synchronize()

                end = time.time()

                inference_time = end - start
                total_inference_time += inference_time
                print(f"Inference time: {total_inference_time/(i+1):.6f} seconds")
                pi_hat_list.append(pi_hat)
                test_y_label.append(y_label)
                A_score_list.append(trendLL)
                true_x_list.append(true_x.detach().cpu())
                rec_d_list.append(rec_d.detach().cpu())
                mu_trend_list.append(mu_trend.detach().cpu())
                scale_tril_list.append(scale_tril.detach().cpu())


        A_score_np = np.concatenate(A_score_list, axis=0).ravel()
        test_y_label_np = np.concatenate(test_y_label, axis=0).ravel()
        gt = test_y_label_np.astype(int)
        resultscoreseq=get_bestF1(gt,A_score_np, PA=False)
        seqthres=resultscoreseq["thres"]
        predseq = (A_score_np >seqthres).astype(int)
        accuracy = accuracy_score(gt, predseq)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, predseq,  average='binary')
        seqout="TrendLike: Accuracy : {:0.2f}, Precision : {:0.2f}, Recall : {:0.2f}, F-score : {:0.2f} ".format(
                accuracy*100, precision*100,
                recall*100, f_score*100)
        roc=roc_auc_score(gt,A_score_np)
        prroc=average_precision_score(gt,A_score_np)
        seqout+="\n\n"+str(roc)
        seqout+="\n\n"+str(prroc)
        print(seqout)
        current_path = os.getcwd()
        today = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        fullpath=current_path+'/results/'
        print("Results saved at"+ f'{fullpath}{today}_result_'+str(self.dataset)+'.txt')

        with open(f'{fullpath}{today}_result_'+str(self.dataset)+'.txt', 'a') as file:
            file.write(seqout)

        return precision, recall, f_score, roc

