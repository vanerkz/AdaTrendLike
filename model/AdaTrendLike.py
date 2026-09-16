import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        return torch.complex(F.relu(z.real), F.relu(z.imag))

class ModReLU(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, z):
        # z: complex tensor (..., D)

        magnitude = torch.abs(z)

        activated = F.relu(magnitude + self.bias.view(1,-1,1))

        phase = z / (magnitude + 1e-8)

        return activated * phase

class FreqCNN(nn.Module):
    def __init__(self,d_model, kernelsize,freqk,outsize,win_size,flayer,cnngroup):
        super(FreqCNN, self).__init__()
        self.paddingk = (kernelsize - 1)//2
        self.paddingfreqk = (freqk - 1) // 2 
        self.d_model=d_model
        self.feature_size=outsize
        self.win_size =win_size
        self.avg = nn.AvgPool1d(kernel_size=kernelsize, stride=1, padding=0)
        self.flayer=flayer
        self.flayers = nn.ModuleList([])
        for _ in range(flayer):
            block = nn.Sequential(
                nn.Conv1d(d_model, d_model, freqk, padding=self.paddingfreqk, groups=cnngroup, bias=False).to(torch.cfloat),
                ComplexReLU()
            )
            self.flayers.append(block)
        finalshape=sum(win_size // (2**i) for i in range(flayer))
        self.out=MLP(finalshape,win_size,finalshape)
        self.trendproj=nn.Linear(win_size,win_size,False)

    def forward(self, inputd, trendtotal,meanin,stdin):
        trend = self.avg(F.pad(inputd, (self.paddingk, self.paddingk), mode='replicate'))
        inputd=inputd-trend
        if trendtotal is None:
            trendtotal = torch.relu(self.trendproj(trend))
        else:
            trendtotal = trendtotal+torch.relu(self.trendproj(trend))
        n_list = [self.win_size // (2**i) for i in range(self.flayer)]  # corresponding time lengths

        freq_list = []
        for i, nlayers in enumerate(self.flayers):
            band = torch.fft.rfft(inputd, n=n_list[i], dim=-1, norm="ortho")
            band = nlayers(band)
            band = torch.fft.irfft(band, n=n_list[i], dim=-1, norm="ortho")
            freq_list.append(band)
        outputd = torch.cat(freq_list, dim=-1)
        outputd=outputd*stdin
        outputd=outputd+meanin
        outputd=self.out(outputd)
        return outputd ,trendtotal

class AdaTrendLike(nn.Module):
    def __init__(self, win_size,c_out, d_model,dlayer,movk,freqk,flayer,cnngroup):
        super(AdaTrendLike, self).__init__()
        self.feature_size=c_out
        self.d_model=d_model
        self.window=win_size
        self.n=win_size
        self.dlayer=dlayer
        self.adadetrendinglayer = nn.ModuleList([
            FreqCNN(d_model, movk, freqk, c_out, win_size, flayer, cnngroup)
            for _ in range(dlayer)
        ])
        d_modellayer=d_model//2
        self.inputembedding =MLP(c_out,d_model,d_modellayer)
        self.output = MLP(d_model,c_out,d_modellayer)
        r =d_model//16
        self.r=r
        self.weightproj = nn.Sequential(
            nn.Conv1d(d_model, d_modellayer, kernel_size=3, padding=1,padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv1d(d_modellayer, 1, kernel_size=3, padding=1,padding_mode="replicate")
        )

        self.offdiagproj = nn.Sequential(
            nn.Conv1d(d_model, r, kernel_size=3, padding=1,padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv1d(r, r*c_out, kernel_size=3, padding=1,padding_mode="replicate")
        )

        self.meanproj = nn.Sequential(
            nn.Conv1d(d_model, d_modellayer, kernel_size=3, padding=1,padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv1d(d_modellayer, c_out, kernel_size=3, padding=1,padding_mode="replicate")
        )

        self.paddingk = (movk - 1)//2
        self.avg = nn.AvgPool1d(kernel_size=movk, stride=1, padding=0)
        self.softplus=nn.Softplus()
        self.linearmean= MLP(win_size,d_model,d_modellayer)
        self.linearvar= MLP(win_size,d_model,d_modellayer)
        self.temp = nn.Parameter(torch.sqrt(torch.tensor(d_model, dtype=torch.float32)))
 
    def standardize_embedding(self, X):
        B,_,_=X.shape
        mean = torch.mean(X,dim=-1, keepdim=True)
        std = torch.std(X, dim=-1,keepdim=True)
        X_standardized=(X-mean)/(std+1e-4)
        meanin=self.linearmean(mean.reshape(B,-1)).unsqueeze(-1)
        stdin=self.linearvar(std.reshape(B,-1)).unsqueeze(-1)
        return X_standardized, meanin, stdin

    def trend_estimation(self, trendtotal):
        B,L,_=trendtotal.shape
        pi_hat = self.weightproj(trendtotal).permute(0,2,1)/self.temp
        pi_hat=pi_hat.squeeze(-1)
        mu_trend = self.meanproj(trendtotal).permute(0,2,1)
        A = self.offdiagproj(trendtotal).permute(0,2,1).reshape(B, self.window, self.feature_size, -1)  
        cov_like =(A @ A.transpose(-1, -2))
        diag = torch.diagonal(cov_like, dim1=-2, dim2=-1)
        new_diag = self.softplus(diag)
        cov_like = cov_like + torch.diag_embed(new_diag - diag)
        scale_tril = torch.tril(cov_like)
        return mu_trend, scale_tril, pi_hat
       
    def forward(self, X):
        true_trend = self.avg(F.pad(X.permute(0,2,1), (self.paddingk,self.paddingk), mode='replicate')).permute(0,2,1)
        X, meanin, stdin = self.standardize_embedding(X)
        D_hat=self.inputembedding(X).permute(0,2,1)
        trendtotal=None
        for _, layer in enumerate(self.adadetrendinglayer):
            res=D_hat
            D_hat,trendtotal= layer(D_hat,trendtotal,meanin,stdin)
            D_hat+=res
        D_hat = self.output(D_hat.permute(0,2,1))
        mu_trend, cov,pi_hat =self.trend_estimation(trendtotal)
        return D_hat, true_trend, mu_trend,cov,pi_hat


class MLP(nn.Module):
    def __init__(self, in_dim,out_dim, hidden_dim,bias=True):
        super().__init__()

        self.mlp=nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim,bias=bias),
            )

    def forward(self, x):
        x = self.mlp(x)
        return x