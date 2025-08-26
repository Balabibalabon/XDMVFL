import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

class OutOfParticipant(Exception):
    pass

def set_seed(seed=500):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
set_seed(42)  # Call the seed function with your desired seed

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
class SHAPMAE(nn.Module):
    def __init__(self, args, active_party):
        super(SHAPMAE, self).__init__()
        '''
        測試維度對防禦效果之影響
        #4096
        self.input_dim = 64 * 8 * 8  
        input_dim = 64 * 8 * 8 # 4096
        '''
        self.args = args
        self.num_participants = args.P
        self.bottomModel_output_dim = args.bottomModel_output_dim
        self.input_dim = args.bottomModel_output_dim * self.num_participants
        self.attack_method = args.AM
        self.defense_method = args.DM
        self.active_party_or_not = args.act
        if self.active_party_or_not:
            self.active_party = active_party
        
        if self.num_participants <8:
            self.s_score_threshold = 70 #VFLIP 4人 TIFS
        elif self.num_participants >=8:
            self.s_score_threshold = 39 #VFLIP 8人 TIFS
        
        self.count = 0
        self.shap_threshold = args.num_class-1 #SHAP
        # x = x.view(x.size(0), -1)

        # Encoder part (原始的 fc1 到 fc3)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, int(self.input_dim/10*9)),
            nn.BatchNorm1d(int(self.input_dim/10*9)),
            nn.ReLU(True),
            
            nn.Linear(int(self.input_dim/10*9), int(self.input_dim/9*8)),
            nn.BatchNorm1d(int(self.input_dim/9*8)),
            nn.ReLU(True),
            
            nn.Linear(int(self.input_dim/9*8), int(self.input_dim/8*7)), 
            nn.BatchNorm1d(int(self.input_dim/8*7)),
            nn.ReLU(True),
            
            nn.Linear(int(self.input_dim/8*7), int(self.input_dim/7*6)),
            nn.BatchNorm1d(int(self.input_dim/7*6)),
            nn.ReLU(True),
            
            nn.Linear(int(self.input_dim/7*6), int(self.input_dim/6*5)), 
            nn.BatchNorm1d(int(self.input_dim/6*5)),
            nn.ReLU(True),
        )
        
        # Decoder part (原始的 fc6 到 fc10)
        self.decoder = nn.Sequential(
            nn.Linear(int(self.input_dim/6*5), int(self.input_dim/7*6)),  # 3906 -> 4032
            nn.BatchNorm1d(int(self.input_dim/7*6)),
            nn.ReLU(True),
            
            nn.Linear(int(self.input_dim/7*6), int(self.input_dim/8*7)),  # 3906 -> 4032
            nn.BatchNorm1d(int(self.input_dim/8*7)),
            nn.ReLU(True),
            
            nn.Linear(int(self.input_dim/8*7), int(self.input_dim/9*8)),  # 3662 -> 3906
            nn.BatchNorm1d(int(self.input_dim/9*8)),
            nn.ReLU(True),
            
            nn.Linear(int(self.input_dim/9*8), int(self.input_dim/10*9)),  # 3906 -> 4032
            nn.BatchNorm1d(int(self.input_dim/10*9)),
            nn.ReLU(True),
            
            nn.Linear(int(self.input_dim/10*9), self.input_dim),              # 4032 -> 4096
            nn.BatchNorm1d(int(self.input_dim))
        )
        
        self.dropout = nn.Dropout(0.1)
        self.train_mode = True
        
        # Model properties
        self.normalize = False
        self.unit = False
        
        # Normalization parameters
        self.mu = 0
        self.std = 1
        
        self.adaptive_thresholds = {}
        
        # 初始化權重
        self.apply(weights_init)
        
        # 設置設備
        self.device = args.device
        self.to(self.device)
        
    def forward(self, x, flagged_passive_dict=None):
        input_tensor = x.to(self.device)
        
        if self.train_mode:
            # Training mode
            if self.normalize:
                input_tensor = (input_tensor - self.mu) / self.std
                orig_input_tensor = input_tensor.clone()
                input_tensor = input_tensor.view(input_tensor.shape[0],-1)
            
            '''
            需要 output 2個結果，分別為 N-1 以及 1-1 的 strategy，以利後面計算 loss
            步驟:
            1. 創建 mask
            2. 讓 input 乘上 mask
            3. 丟入 MAE
            '''
            """
            前向傳播邏輯
            
            Args:
                x (torch.Tensor): 輸入張量，包含所有參與者的特徵
            
            Returns:
                dict: 包含 N-1 和 1-1 策略的重建結果
            """
            
            # N-1 策略
            masked_input_n_minus_one, mask_n_minus_one, target_mask_n_minus_one = self.create_mask(x, mode='n_minus_one')
            latent_n_minus_one = self.encoder(masked_input_n_minus_one.view(x.size(0), -1))
            output_n_minus_one = self.decoder(latent_n_minus_one)
            
            # Safely modify output based on target mask
            if target_mask_n_minus_one is not None:
                # Only modify the parts of the output corresponding to the masked participant
                non_zero_indices = (target_mask_n_minus_one.view(-1) == 1).nonzero(as_tuple=True)[0]
                if len(non_zero_indices) > 0:
                    output_n_minus_one_masked = output_n_minus_one.clone()
                    output_n_minus_one_masked.view(-1)[non_zero_indices] = 0
                else:
                    output_n_minus_one_masked = output_n_minus_one
            else:
                output_n_minus_one_masked = output_n_minus_one
            
            # 1-1 策略
            masked_input_one_to_one, mask_one_to_one, target_mask_one_to_one = self.create_mask(x, mode='one_to_one')
            latent_one_to_one = self.encoder(masked_input_one_to_one.view(x.size(0), -1))
            output_one_to_one = self.decoder(latent_one_to_one)
            
            # Safely modify output for one-to-one strategy
            if target_mask_one_to_one is not None:
                non_zero_indices = (target_mask_one_to_one.view(-1) == 1).nonzero(as_tuple=True)[0]
                if len(non_zero_indices) > 0:
                    output_one_to_one_masked = output_one_to_one.clone()
                    output_one_to_one_masked.view(-1)[non_zero_indices] = 0
                else:
                    output_one_to_one_masked = output_one_to_one
            else:
                output_one_to_one_masked = output_one_to_one
            
            # 原始輸入的編碼和解碼
            '''
            測試 dropout 比例影響，以及是否需要 dropout
            '''
            input_tensor = self.dropout(input_tensor)
            latent = self.encoder(input_tensor)
            output = self.decoder(latent)
            
            if self.normalize:
                output = (output * self.std) + self.mu
            
            return {
                'output_n_minus_one': output_n_minus_one,
                'mask_n_minus_one': mask_n_minus_one,
                'target_mask_n_minus_one': target_mask_n_minus_one,
                'output_one_to_one': output_one_to_one,
                'mask_one_to_one': mask_one_to_one,
                'target_mask_one_to_one': target_mask_one_to_one,
                'original_input': x,
                'output': output,
            }
        
        else:
            self.count+=1
            # Evaluation mode
            if self.count == 1:
                # print("Eval mode")
                pass
            if self.normalize:
                input_tensor = (input_tensor - self.mu) / self.std #標準化
                orig_input_tensor = input_tensor.clone() 
                input_tensor = input_tensor.view(input_tensor.shape[0],-1)
            
            '''
            defense_method 為 SHAP 
            1. 根據 counter_dict 獲取每個參與者在每個sample shapley value<0的class數量，進行mask製作
            1. 每個 sample 都會判斷哪幾個參與者在預測的結果外的label shapley value 皆為負
            2. 將其在該 sample 的輸入用遮罩遮起來，進行還原
            '''
            # 根據 flagged_passive_dict 對 input_tensor 建立遮罩
            SHAP_MAE_mask = torch.ones_like(input_tensor)

            for i in range(input_tensor.shape[0]):  # 對每一筆 sample
                # if i not in flagged_passive_dict:
                #     continue
                for p in flagged_passive_dict[i]:  # 對於該 sample 中的每一個潛在攻擊者參與者
                    if self.active_party_or_not and p == self.active_party:
                        continue
                    start_idx = p * self.bottomModel_output_dim
                    end_idx = (p + 1) * self.bottomModel_output_dim
                    SHAP_MAE_mask[i, start_idx:end_idx] = 0
            
            # 僅對被檢測到的部分進行重構：先將原始輸入乘上 MAE_mask (保留 benign 部分)                
            latent = self.encoder(input_tensor * SHAP_MAE_mask)
            reconstructed = self.decoder(latent)
            if self.normalize:
                reconstructed = (reconstructed * self.std) + self.mu

            # 最終輸出：被檢測到的部分 (mask=0) 使用 reconstructed，其他部分則保留原始輸入
            final_output = input_tensor * SHAP_MAE_mask + reconstructed * (1 - SHAP_MAE_mask)
            return final_output
    
    def create_mask(self, input_tensor, mode='n_minus_one'):
        """
        創建遮罩
        
        Args:
            input_tensor (torch.Tensor): 輸入張量
            mode (str): 遮罩模式 - 'n_minus_one' 或 'one_to_one'
        
        Returns:
            torch.Tensor: 遮罩後的輸入張量
        """
        
        original_shape = input_tensor.shape
        batch_size = input_tensor.size(0)
        #這邊分割方法要視每個模型設定而定
        Embedding_vector_dim_size = input_tensor.size(1)
        
        Embedding_vector_dim_per_participant = Embedding_vector_dim_size // self.num_participants

        # 初始化遮罩和目標遮罩
        target_mask = torch.zeros_like(input_tensor)

        if mode == 'n_minus_one':
            # 隨機選擇一個參與者
            mask = torch.ones_like(input_tensor)
            if self.args.act:
                '''
                可以考慮將active aprty固定為參考對象
                '''
                masked_participant = 0
            else:
                masked_participant = torch.randint(0, self.num_participants, (1,)).item()
            for b in range(batch_size):
                start_idx = masked_participant * Embedding_vector_dim_per_participant
                end_idx = (masked_participant + 1) * Embedding_vector_dim_per_participant
                # print("n-1 to 1 mask 維度:",start_idx,"-",end_idx-1)
                mask[b, start_idx:end_idx] = 0  
                target_mask[b, start_idx:end_idx] = 1
            # print(f"n-1被遮起來的人是:{masked_participant}")

        elif mode == 'one_to_one':
            # 隨機選擇兩個不同的參與者
            mask = torch.zeros_like(input_tensor)
            if self.args.act:
                '''
                可以考慮將active aprty固定為參考對象
                '''
                participant_pairs = torch.tensor([1,0])
            else:
                participant_pairs = torch.randperm(self.num_participants)[:2]
            for b in range(batch_size):
                # 保留第一個參與者
                start_idx_1 = participant_pairs[0] * Embedding_vector_dim_per_participant
                end_idx_1 = (participant_pairs[0] + 1) * Embedding_vector_dim_per_participant
                mask[b, start_idx_1:end_idx_1] = 1

                # print("1 to 1 保留者1 mask 維度:",start_idx_1,"-",end_idx_1-1)

                # 設置目標遮罩為第二個參與者
                start_idx_2 = participant_pairs[1] * Embedding_vector_dim_per_participant
                end_idx_2 = (participant_pairs[1] + 1) * Embedding_vector_dim_per_participant
                target_mask[b, start_idx_2:end_idx_2] = 1

                # print("1 to 1 保留者2 mask 維度:",start_idx_2,"-",end_idx_2-1)
            # print(f"1-1被遮起來的人是:{participant_pairs}")

        else:
            raise ValueError("Unsupported mode. Use 'n_minus_one' or 'one_to_one'.")

        # 確保遮罩形狀一致
        mask = mask.view(original_shape)
        target_mask = target_mask.view(original_shape)

        return input_tensor * mask, mask, target_mask
    
    def set_mode(self, train=True):
        """設置模型模式"""
        self.train_mode = train
    
    def set_Normalization(self,mu,std):
        self.mu = mu
        self.std = std
        # print(f"目前 MAE 的mu為:{self.mu}，std為:{self.std}")
    
    def set_adaptive_threshold(self, Htrain):
        print(f"正在計算每個參與者的 adaptive threshold")
        scores = {i: [] for i in range(self.num_participants)}

        with torch.no_grad():
            for each_emb in Htrain:  # each_emb shape = [D], 1個sample的concat embedding
                # 這裡要遍歷 i=0..(P-1)，用 "遮 i 留 j" 方式計算 s_j->i
                for i in range(self.num_participants):
                    # step a. 製作只留 j≠i 的 mask
                    mask = torch.zeros_like(each_emb)
                    # 這裡示意把 i 那段置 0，所有 j≠i 那段置 1
                    # or 反之 if your code requires, 請確保實際意義正確
                    for j in range(self.num_participants):
                        if j != i:
                            mask[j*self.bottomModel_output_dim : (j+1)*self.bottomModel_output_dim] = 1

                    # step b. feed to MAE
                    x_in = (each_emb * mask).unsqueeze(0)  # shape [1, D]
                    if self.normalize:
                        x_in = (x_in - self.mu) / self.std
                    recon = self.decoder(self.encoder(x_in)) # shape [1, D]

                    # step c. 取出 recon 對應 i 區段、跟真實 i 區段做 L2-norm
                    hi_true  = each_emb[i*self.bottomModel_output_dim : (i+1)*self.bottomModel_output_dim]
                    hi_recon = recon[0, i*self.bottomModel_output_dim : (i+1)*self.bottomModel_output_dim]
                    s_i = (hi_recon - hi_true).norm(2).item()

                    # 收集起來
                    scores[i].append(s_i)

        # 事後算每個 i 的 mean, std
        rho = 2.0  # 例如
        for i in range(self.num_participants):
            s_tensor = torch.tensor(scores[i])
            mu_i = s_tensor.mean().item()
            sigma_i = s_tensor.std().item()
            self.adaptive_thresholds[i] = mu_i + rho * sigma_i
    
    def loss(self, MAE_output):
        MAE_loss_class = VerticalFLMAELoss()
        MAE_loss = MAE_loss_class(MAE_output)
        return MAE_loss

class VerticalFLMAELoss(nn.Module):
    def __init__(self, alpha=1, beta=0.1, gamma=0.1):
        """
        垂直聯邦學習 MAE 的 Loss 函數s
        
        Args:
            alpha (float): N-1 策略的權重
            beta (float): 1-1 策略的權重
        """
        super(VerticalFLMAELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def forward(self, model_output):
        """
        計算 Loss
        
        Args:
            model_output (dict): MAE 模型的輸出
        
        Returns:
            torch.Tensor: 總 Loss
        """
        output_n_minus_one = model_output['output_n_minus_one']
        mask_n_minus_one = model_output['mask_n_minus_one']
        target_mask_n_minus_one = model_output['target_mask_n_minus_one']
        output_one_to_one = model_output['output_one_to_one']
        mask_one_to_one = model_output['mask_one_to_one']
        target_mask_one_to_one = model_output['target_mask_one_to_one']
        original_input = model_output['original_input']
        
        B = mask_n_minus_one.size()[0]
        mask_n_minus_one = mask_n_minus_one.view(B,-1)
        target_mask_n_minus_one = target_mask_n_minus_one.view(B,-1)
        
        mask_one_to_one = mask_one_to_one.view(B,-1)
        target_mask_one_to_one = target_mask_one_to_one.view(B,-1)
        
        original_input = original_input.view(B,-1)
        
        loss_n_minus_one = F.mse_loss(
            output_n_minus_one[mask_n_minus_one == 0], 
            original_input[target_mask_n_minus_one == 1]
        )

        loss_one_to_one = F.mse_loss(
            output_one_to_one[target_mask_one_to_one == 1], 
            original_input[target_mask_one_to_one == 1]
        )
        total_loss = self.alpha * loss_n_minus_one + self.beta * loss_one_to_one
        
        return total_loss

class MAE(nn.Module):
    def __init__(self, args, active_party):
        super(MAE, self).__init__()
        '''
        測試維度對防禦效果之影響
        #4096
        self.input_dim = 64 * 8 * 8  
        input_dim = 64 * 8 * 8 # 4096
        '''
        self.args = args
        self.num_participants = args.P
        self.bottomModel_output_dim = args.bottomModel_output_dim
        self.input_dim = args.bottomModel_output_dim * self.num_participants
        self.attack_method = args.AM
        self.defense_method = args.DM
        self.active_party_or_not = args.act
        if self.active_party_or_not:
            self.active_party = active_party

        self.cantWork = 1 
        if args.AM =="VILLIAN":
            if args.P == 8:
                self.s_score_threshold = 8
            elif args.P == 4:
                self.s_score_threshold = 7
            else:
                self.s_score_threshold = self.cantWork
        elif args.AM == "TIFS":
            if args.P == 8:
                self.s_score_threshold = 10
            elif args.P == 4:
                self.s_score_threshold = 20
            else:
                self.s_score_threshold = self.cantWork
            
         
        
        self.count = 0
        self.shap_threshold = args.num_class-1 #SHAP
        # x = x.view(x.size(0), -1)

        # Encoder part (原始的 fc1 到 fc3)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, int(self.input_dim/10*9)),
            nn.BatchNorm1d(int(self.input_dim/10*9)),
            nn.ReLU(True),
            
            nn.Linear(int(self.input_dim/10*9), int(self.input_dim/9*8)),
            nn.BatchNorm1d(int(self.input_dim/9*8)),
            nn.ReLU(True),
            
            nn.Linear(int(self.input_dim/9*8), int(self.input_dim/8*7)), 
            nn.BatchNorm1d(int(self.input_dim/8*7)),
            nn.ReLU(True),
            
            nn.Linear(int(self.input_dim/8*7), int(self.input_dim/7*6)),
            nn.BatchNorm1d(int(self.input_dim/7*6)),
            nn.ReLU(True),
            
            nn.Linear(int(self.input_dim/7*6), int(self.input_dim/6*5)), 
            nn.BatchNorm1d(int(self.input_dim/6*5)),
            nn.ReLU(True),
        )
        
        # Decoder part (原始的 fc6 到 fc10)
        self.decoder = nn.Sequential(
            
            nn.Linear(int(self.input_dim/6*5), int(self.input_dim/7*6)),  # 3906 -> 4032
            nn.BatchNorm1d(int(self.input_dim/7*6)),
            nn.ReLU(True),
            
            nn.Linear(int(self.input_dim/7*6), int(self.input_dim/8*7)),  # 3906 -> 4032
            nn.BatchNorm1d(int(self.input_dim/8*7)),
            nn.ReLU(True),
            
            nn.Linear(int(self.input_dim/8*7), int(self.input_dim/9*8)),  # 3662 -> 3906
            nn.BatchNorm1d(int(self.input_dim/9*8)),
            nn.ReLU(True),
            
            nn.Linear(int(self.input_dim/9*8), int(self.input_dim/10*9)),  # 3906 -> 4032
            nn.BatchNorm1d(int(self.input_dim/10*9)),
            nn.ReLU(True),
            
            nn.Linear(int(self.input_dim/10*9), self.input_dim),              # 4032 -> 4096
            nn.BatchNorm1d(int(self.input_dim))
        )
        
        self.dropout = nn.Dropout(0.1)
        self.train_mode = True
        
        # Model properties
        self.normalize = False
        self.unit = False
        # self.unit = True
        
        # Normalization parameters
        self.mu = 0
        self.std = 1
        
        self.adaptive_thresholds = {}
        
        # 初始化權重
        self.apply(weights_init)
        
        # 設置設備
        self.device = args.device
        self.to(self.device)
        
    def forward(self, x, counter_dict=None):
        input_tensor = x.to(self.device)
        # print("MAE input tensor size:", input_tensor.shape)
        
        if self.train_mode:
            # Training mode
            if self.normalize:
                input_tensor = (input_tensor - self.mu) / self.std
                orig_input_tensor = input_tensor.clone()
                input_tensor = input_tensor.view(input_tensor.shape[0],-1)
            
            '''
            需要 output 2個結果，分別為 N-1 以及 1-1 的 strategy，以利後面計算 loss
            步驟:
            1. 創建 mask
            2. 讓 input 乘上 mask
            3. 丟入 MAE
            '''
            """
            前向傳播邏輯
            
            Args:
                x (torch.Tensor): 輸入張量，包含所有參與者的特徵
            
            Returns:
                dict: 包含 N-1 和 1-1 策略的重建結果
            """
            
            # N-1 策略
            masked_input_n_minus_one, mask_n_minus_one, target_mask_n_minus_one = self.create_mask(x, mode='n_minus_one')
            latent_n_minus_one = self.encoder(masked_input_n_minus_one.view(x.size(0), -1))
            output_n_minus_one = self.decoder(latent_n_minus_one)
            
            # Safely modify output based on target mask
            if target_mask_n_minus_one is not None:
                # Only modify the parts of the output corresponding to the masked participant
                non_zero_indices = (target_mask_n_minus_one.view(-1) == 1).nonzero(as_tuple=True)[0]
                if len(non_zero_indices) > 0:
                    output_n_minus_one_masked = output_n_minus_one.clone()
                    output_n_minus_one_masked.view(-1)[non_zero_indices] = 0
                else:
                    output_n_minus_one_masked = output_n_minus_one
            else:
                output_n_minus_one_masked = output_n_minus_one
            
            # 1-1 策略
            masked_input_one_to_one, mask_one_to_one, target_mask_one_to_one = self.create_mask(x, mode='one_to_one')
            latent_one_to_one = self.encoder(masked_input_one_to_one.view(x.size(0), -1))
            output_one_to_one = self.decoder(latent_one_to_one)
            
            # Safely modify output for one-to-one strategy
            if target_mask_one_to_one is not None:
                non_zero_indices = (target_mask_one_to_one.view(-1) == 1).nonzero(as_tuple=True)[0]
                if len(non_zero_indices) > 0:
                    output_one_to_one_masked = output_one_to_one.clone()
                    output_one_to_one_masked.view(-1)[non_zero_indices] = 0
                else:
                    output_one_to_one_masked = output_one_to_one
            else:
                output_one_to_one_masked = output_one_to_one
            
            # 原始輸入的編碼和解碼
            '''
            測試 dropout 比例影響，以及是否需要 dropout
            '''
            input_tensor = self.dropout(input_tensor)
            latent = self.encoder(input_tensor)
            output = self.decoder(latent)
            
            if self.normalize:
                output = (output * self.std) + self.mu
            
            return {
                'output_n_minus_one': output_n_minus_one,
                'mask_n_minus_one': mask_n_minus_one,
                'target_mask_n_minus_one': target_mask_n_minus_one,
                'output_one_to_one': output_one_to_one,
                'mask_one_to_one': mask_one_to_one,
                'target_mask_one_to_one': target_mask_one_to_one,
                'original_input': x,
                'output': output,
            }
        
        else:
            self.count+=1
            # Evaluation mode
            if self.count == 1:
                print("Eval mode")
                
                '''
                test
                '''
                print("—— MAE Eval 1st batch ——")
                print("adaptive_thresholds =", list(self.adaptive_thresholds.items()))
                
                if self.count == 1:
                    print("Eval mode")
                    print("⟹ 已載入門檻 keys：", list(self.adaptive_thresholds.keys())[:5], "…")
                    print("⟹ 例如 t_0, t_1, t_2 =", 
                        self.adaptive_thresholds.get(0, None),
                        self.adaptive_thresholds.get(1, None),
                        self.adaptive_thresholds.get(2, None))
                    
            if self.normalize:
                input_tensor = (input_tensor - self.mu) / self.std #標準化
                orig_input_tensor = input_tensor.clone() 
                input_tensor = input_tensor.view(input_tensor.shape[0],-1)
                
            if self.defense_method == "VFLIP":
                orig_input_tensor = input_tensor.clone() 
                # Calculate reconstruction scores for each party
                s_score = torch.zeros([input_tensor.shape[0], self.num_participants, self.num_participants]).to(self.device) #建立每個 sample 每個參與者對每個參與者的 s_score 計算 
                # print("s_score size:", s_score.shape)
                for i in range(self.num_participants):
                    mask = torch.zeros_like(orig_input_tensor).to(self.device) #建立遮住某個參與者輸入的遮罩
                    mask[:, i*self.bottomModel_output_dim:(i+1)*self.bottomModel_output_dim] = 1

                    mask_input = (mask * orig_input_tensor).view(input_tensor.shape)
                    latent = self.encoder(mask_input)
                    
                    recon = self.decoder(latent)
                    
                    for j in range(self.num_participants):
                        if i == j:
                            continue
                        # print(f"j為:{j}")
                        norm_for_j = torch.norm(
                            recon[ :, j*self.bottomModel_output_dim:(j+1)*self.bottomModel_output_dim] - 
                            input_tensor[ :, j*self.bottomModel_output_dim:(j+1)*self.bottomModel_output_dim],
                            dim=1
                        ) 
                        
                        for k in range(norm_for_j.shape[0]): 
                            s_score[k,i,j] = norm_for_j[k]
                
                if self.count == 1:                      # 只印第一個 batch
                    # 取第一筆樣本第一位 passive 的 s-score 作參考
                    sample0_scores = s_score[0]          # shape [P, P]
                    print("示例 s_score[0] matrix:\n", sample0_scores.cpu().numpy())

                '''
                原先寫死 voting 閾值
                # Create mask based on voting
                voting_matrix = torch.sum((s_score > self.s_score_threshold).int(), dim=1) # 根據每筆 sample 統計哪幾個參與者在該 sample 大於閾值應該被替換
                '''
                
                # ---------- 依 adaptive_thresholds 判斷是否超標 ----------
                if hasattr(self, "adaptive_thresholds") and \
                len(self.adaptive_thresholds) == self.num_participants:
                    thr_vec = torch.tensor([self.adaptive_thresholds[p]
                                            for p in range(self.num_participants)],
                                        device=self.device).view(1, 1, -1)      # shape 1×1×P
                    exceed = (s_score > thr_vec)        # bool [B,i,j] 逐人門檻
                else:                                   # 尚未計算門檻時退回全局值
                    exceed = (s_score > self.s_score_threshold)

                # ---------- 多數決 ----------
                voting_matrix = exceed.int().sum(dim=1)    # shape [B, P]

                MAE_mask = torch.ones_like(input_tensor)
                # print("mask dim:",mask.shape)
                
                '''
                測試 mask 到底有沒有用
                '''
                # count = 0
                if self.count < 21:
                    s_score = s_score.reshape(s_score.shape[0],-1).cpu().numpy()
                    f = open("clean_sscore_record.txt","a")
                    np.savetxt(f,s_score)
                    f.close()
                    
                    voting_matrix = voting_matrix.cpu().numpy()
                    f = open("clean_voting_record.txt","a")
                    np.savetxt(f,voting_matrix)
                    f.close()
                else: 
                    s_score = s_score.reshape(s_score.shape[0],-1).cpu().numpy()
                    f = open("poisoned_sscore_record.txt","a")
                    np.savetxt(f,s_score)
                    f.close()
                    voting_matrix = voting_matrix.cpu().numpy()
                    f = open("poisoned_voting_record.txt","a")
                    np.savetxt(f,voting_matrix)
                    f.close()
                
                
                # print(voting_matrix.shape)
                testtesttest = 0
                for i in range(voting_matrix.shape[0]): # i 代表 sample number
                    for j in range(voting_matrix.shape[1]): # j 代表在該 sample number 中每個參與者被投票的次數
                        # print(voting_matrix[i,j].item())
                        '''大於等於1/2 是我調的設置，原先只有大於1/2，但如此一來永遠偵測不到
                        if voting_matrix[i,j].item() >= int(self.num_participants/2): '''
                        # if self.active_party_or_not==True and j == self.active_party:
                        #     continue
                        if voting_matrix[i,j].item() > int(self.num_participants/2): 
                            testtesttest+=1
                            # print(voting_matrix[i,j].item())
                            # count+=1
                            MAE_mask[i, j*self.bottomModel_output_dim:(j+1)*self.bottomModel_output_dim] = 0 # 建立遮罩，屏蔽該參與者在該 sample 輸入，讓 MAE 緩解
                print(f"在這個 batch 中 總共有{testtesttest}次被抓到")
                if torch.all(MAE_mask == 1):
                    return input_tensor  # 直接返回未經修改的向量
                
                # 僅對被檢測到的部分進行重構：先將原始輸入乘上 MAE_mask (保留 benign 部分)
                latent = self.encoder(input_tensor * MAE_mask)
                reconstructed = self.decoder(latent)
                if self.normalize:
                    reconstructed = (reconstructed * self.std) + self.mu

                # 最終輸出：被檢測到的部分 (mask=0) 使用 reconstructed，其他部分則保留原始輸入
                final_output = input_tensor * MAE_mask + reconstructed * (1 - MAE_mask)
                return final_output

            
            elif self.defense_method == "SHAP":
                '''
                defense_method 為 SHAP 
                1. 根據 counter_dict 獲取每個參與者在每個sample shapley value<0的class數量，進行mask製作
                1. 每個 sample 都會判斷哪幾個參與者在預測的結果外的label shapley value 皆為負
                2. 將其在該 sample 的輸入用遮罩遮起來，進行還原
                '''
                MAE_mask = torch.ones_like(input_tensor)
                for p in range(self.num_participants):
                    if self.active_party_or_not==True and p == self.active_party:
                        continue
                    else:
                        sample_mask = (counter_dict[p] >= self.shap_threshold)
                        for i in range(input_tensor.shape[0]):
                            if sample_mask[i]:
                                MAE_mask[i, p*self.bottomModel_output_dim:(p+1)*self.bottomModel_output_dim] = 0
                            
                latent = self.encoder(input_tensor * MAE_mask)
                reconstructed = self.decoder(latent)
                if self.normalize:
                    reconstructed = (reconstructed * self.std) + self.mu

                # 最終輸出：被檢測到的部分 (mask=0) 使用 reconstructed，其他部分則保留原始輸入
                final_output = input_tensor * MAE_mask + reconstructed * (1 - MAE_mask)
                return final_output
                    
    
    def create_mask(self, input_tensor, mode='n_minus_one'):
        """
        創建遮罩
        
        Args:
            input_tensor (torch.Tensor): 輸入張量
            mode (str): 遮罩模式 - 'n_minus_one' 或 'one_to_one'
        
        Returns:
            torch.Tensor: 遮罩後的輸入張量
        """
        
        original_shape = input_tensor.shape
        batch_size = input_tensor.size(0)
        #這邊分割方法要視每個模型設定而定
        Embedding_vector_dim_size = input_tensor.size(1)
        
        Embedding_vector_dim_per_participant = Embedding_vector_dim_size // self.num_participants

        # 初始化遮罩和目標遮罩
        target_mask = torch.zeros_like(input_tensor)

        if mode == 'n_minus_one':
            # 隨機選擇一個參與者
            mask = torch.ones_like(input_tensor)
            if self.args.act:
                '''
                可以考慮將active aprty固定為參考對象
                '''
                masked_participant = 0
            else:
                masked_participant = torch.randint(0, self.num_participants, (1,)).item()
            for b in range(batch_size):
                start_idx = masked_participant * Embedding_vector_dim_per_participant
                end_idx = (masked_participant + 1) * Embedding_vector_dim_per_participant
                # print("n-1 to 1 mask 維度:",start_idx,"-",end_idx-1)
                mask[b, start_idx:end_idx] = 0  
                target_mask[b, start_idx:end_idx] = 1
            # print(f"n-1被遮起來的人是:{masked_participant}")

        elif mode == 'one_to_one':
            # 隨機選擇兩個不同的參與者
            mask = torch.zeros_like(input_tensor)
            if self.args.act:
                '''
                可以考慮將active aprty固定為參考對象
                '''
                participant_pairs = torch.tensor([1,0])
            else:
                participant_pairs = torch.randperm(self.num_participants)[:2]
            for b in range(batch_size):
                # 保留第一個參與者
                start_idx_1 = participant_pairs[0] * Embedding_vector_dim_per_participant
                end_idx_1 = (participant_pairs[0] + 1) * Embedding_vector_dim_per_participant
                mask[b, start_idx_1:end_idx_1] = 1

                # print("1 to 1 保留者1 mask 維度:",start_idx_1,"-",end_idx_1-1)

                # 設置目標遮罩為第二個參與者
                start_idx_2 = participant_pairs[1] * Embedding_vector_dim_per_participant
                end_idx_2 = (participant_pairs[1] + 1) * Embedding_vector_dim_per_participant
                target_mask[b, start_idx_2:end_idx_2] = 1

                # print("1 to 1 保留者2 mask 維度:",start_idx_2,"-",end_idx_2-1)
            # print(f"1-1被遮起來的人是:{participant_pairs}")

        else:
            raise ValueError("Unsupported mode. Use 'n_minus_one' or 'one_to_one'.")

        # 確保遮罩形狀一致
        mask = mask.view(original_shape)
        target_mask = target_mask.view(original_shape)

        return input_tensor * mask, mask, target_mask
        
    def set_mode(self, train=True):
        """設置模型模式"""
        self.train_mode = train
    
    def set_Normalization(self,mu,std):
        self.mu = mu
        self.std = std
        # print(f"目前 MAE 的mu為:{self.mu}，std為:{self.std}")
    
    def set_adaptive_threshold(self, Htrain):
        print(f"正在計算每個參與者的 adaptive threshold")
        scores = {i: [] for i in range(self.num_participants)}

        with torch.no_grad():
            for each_emb in Htrain:  # each_emb shape = [D], 1個sample的concat embedding
                # 這裡要遍歷 i=0..(P-1)，用 "遮 i 留 j" 方式計算 s_j->i
                for i in range(self.num_participants):
                    # step a. 製作只留 j≠i 的 mask
                    mask = torch.zeros_like(each_emb)
                    # 這裡示意把 i 那段置 0，所有 j≠i 那段置 1
                    # or 反之 if your code requires, 請確保實際意義正確
                    for j in range(self.num_participants):
                        if j != i:
                            mask[j*self.bottomModel_output_dim : (j+1)*self.bottomModel_output_dim] = 1

                    # step b. feed to MAE
                    x_in = (each_emb * mask).unsqueeze(0)  # shape [1, D]
                    if self.normalize:
                        x_in = (x_in - self.mu) / self.std
                    recon = self.decoder(self.encoder(x_in)) # shape [1, D]

                    # step c. 取出 recon 對應 i 區段、跟真實 i 區段做 L2-norm
                    hi_true  = each_emb[i*self.bottomModel_output_dim : (i+1)*self.bottomModel_output_dim]
                    hi_recon = recon[0, i*self.bottomModel_output_dim : (i+1)*self.bottomModel_output_dim]
                    s_i = (hi_recon - hi_true).norm(2).item()

                    # 收集起來
                    scores[i].append(s_i)

        # 事後算每個 i 的 mean, std
        rho = 2.0  # 例如
        for i in range(self.num_participants):
            s_tensor = torch.tensor(scores[i])
            mu_i = s_tensor.mean().item()
            sigma_i = s_tensor.std().item()
            self.adaptive_thresholds[i] = mu_i + rho * sigma_i
    
    def loss(self, MAE_output):
        MAE_loss_class = VerticalFLMAELoss()
        MAE_loss = MAE_loss_class(MAE_output)
        return MAE_loss
    
# -------------------------------------------------------------
# ===  CoPur AutoEncoder + 防禦模組  ===========================
# -------------------------------------------------------------
import torch, torch.nn as nn, torch.nn.functional as F

class CoPurEncoder(nn.Module):
    def __init__(self, in_dim, hid=2048, code=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, code)          # no activation
        )
    def forward(self, x): return self.net(x)

class CoPurDecoder(nn.Module):
    def __init__(self, out_dim, hid=2048, code=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(code, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, out_dim)       # no activation
        )
    def forward(self, z): return self.net(z)

class CoPurDefense:
    """
    只需兩步：
        1) train_autoencoder(H_train_tensor)
        2) clean_h = purify(raw_h, observed_mask)
    """
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.num_participants = args.P
        self.bottomModel_output_dim = args.bottomModel_output_dim
        self.input_dim = args.bottomModel_output_dim * self.num_participants
        self.enc  = CoPurEncoder(self.input_dim).to(self.device)
        self.dec  = CoPurDecoder(self.input_dim).to(self.device)
        self.mu = self.std = None
        self.delta = None          # 之後自動估

    # ---------- (1) 離線訓練 ----------
    def train_autoencoder(self, H_train):
        # H = torch.tensor(H_train, dtype=torch.float32, device=self.device)
        H = torch.as_tensor(H_train, dtype=torch.float32, device=self.device).clone().detach()
        self.mu  = H.mean(0, keepdim=True)
        self.std = H.std (0, keepdim=True) + 1e-6
        Hn = (H - self.mu)/self.std

        opt = torch.optim.Adam(list(self.enc.parameters())+
                               list(self.dec.parameters()), lr=1e-3)
        for ep in range(20):
            perm = torch.randperm(Hn.size(0))
            for i in range(0, Hn.size(0), 256):
                b = Hn[perm[i:i+256]]
                recon = self.dec(self.enc(b))
                loss  = F.mse_loss(recon, b)
                opt.zero_grad(); loss.backward(); opt.step()

        # 自動設定 δ：P95 重建誤差
        with torch.no_grad():
            err = (self.dec(self.enc(Hn))-Hn).pow(2).sum(1).sqrt()
        delta_cfg = getattr(self.args, "copur_delta", 0.0)
        self.delta = err.quantile(0.95).item() if delta_cfg == 0.0 else delta_cfg

    # ---------- (2) 推論淨化 ----------
    def _norm  (self, x): return (x-self.mu)/self.std
    def _denorm(self, x): return x*self.std + self.mu

    def purify(self, raw_h, observed_mask=None):
        # ------------- 開啟 grad -------------
        with torch.enable_grad():        # ★ 新增整個區塊
            B, PD = raw_h.shape
            P     = self.num_participants
            D     = PD // P

            if observed_mask is None:
                observed_mask = torch.ones((B, P), dtype=torch.bool, device=raw_h.device)
            else:
                observed_mask = observed_mask.to(raw_h.device)

            h  = self._norm(raw_h)
            l  = h.clone().detach().requires_grad_(True)

            opt   = torch.optim.Adam([l], lr=1e-2)
            mask3 = observed_mask.unsqueeze(-1).float()

            for _ in range(self.args.copur_iter):
                recon     = self.dec(self.enc(l))
                loss_obs  = ((l-h).view(B, P, D) * mask3).norm(dim=2).sum(1).mean()
                loss_man  = (recon - l).pow(2).sum(1).sqrt().mean()
                loss      = loss_obs + self.args.copur_tau * loss_man
                opt.zero_grad(); loss.backward(); opt.step()

            clean = self._denorm(self.dec(self.enc(l.detach())))
            return clean