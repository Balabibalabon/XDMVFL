import data
import time
import random
import torch
import numpy as np
import torchvision.transforms as transforms

class NeedMoreEpochs(Exception):
    pass

def get_start_poison_epoch(args):
    if args.AM in ["TIFS"]:
        s_p_e = args.epochs-20
    elif args.AM in ["VILLIAN","BadVFL"]:
        s_p_e = 5
    if s_p_e <= 0:
        raise NeedMoreEpochs("Please make sure the number of epochs more than 40")
    return s_p_e

# Attack.py 〈最前方或 utils 區〉
def _split_for_parties(x: torch.Tensor, P: int, D: str):
    """
    將輸入切成 P 份：
      • 影像 (B,C,H,W) → 沿 H 均分；餘數併入最後一份。
      • 表格 (B,F)     → 沿 F 均分；餘數併入最後一份。
    回傳 list[Tensor]，長度必為 P；不重複、不遺漏像素／特徵，也不多回傳餘數切片。
    """
    if D in ["CIFAR10", "CINIC10"]:           # -------- 影像 ----------
        H = x.size(2)
        piece = H // P                        # 基本切片高度
        chunks = list(torch.split(x, piece, dim=2))  # 可能得到 P 或 P+1 塊
        if len(chunks) > P:                   # 有餘數塊
            chunks[-2] = torch.cat((chunks[-2], chunks[-1]), dim=2)  # 併到最後一人
            chunks = chunks[:-1]              # 移除空殼
        return chunks                         # len == P，最後一塊高 = piece + r

    elif D in ["BM","NUSWIDE"]:                          
        # -------- 表格 ----------
        # 新增：有 feature_slices 就按它切
        if hasattr(data, "FEATURE_SLICES") and data.FEATURE_SLICES is not None:
            # FEATURE_SLICES 例： [11, 11, 11, 10]
            lengths = data.FEATURE_SLICES
            assert len(lengths) == P, "FEATURE_SLICES 長度應等於參與者數"
            splits = torch.split(x, lengths, dim=1)
            return list(splits)
        
        # ↓ 下面是 fallback 舊邏輯（平均切）
        F = x.size(1)
        piece = F // P
        # 前 P-1 位各拿 piece 個特徵
        slices = [x[:, i*piece:(i+1)*piece] for i in range(P-1)]
        # 最後一位拿剩下所有特徵 (piece + r)
        slices.append(x[:, (P-1)*piece : ])
        return slices                         # len == P，同樣無餘片

    else:
        raise RuntimeError(f"unsupported dataset type {D!r}")
        
class TIFS_attack():
    def __init__(self,args):
        self.D = args.D
        
        pr = getattr(args, "poison_ratio", None)
        if pr is not None:
            if self.D == "CIFAR10":
                self.poison_sample_num = int(50000*pr)
            elif self.D == "CINIC10":
                self.poison_sample_num = int(90000*pr)
            elif self.D == "BM":
                self.poison_sample_num = int(8929*pr)
        else:
            self.poison_sample_num = 500
        # self.poison_sample_num = 500
        
        
    def get_design_vec(self, model, args, steal_set, attacker_list):
        import generate_vec as gv
        import cal_centers as cc
        import search_vec as sv
        
        print("正在製作新一輪 TIFS 後門 embedding vector")
        attacker_clean_vec_dict = gv.generate_target_clean_vecs(model, steal_set, args, attacker_list)
        
        attacker_selected = self.filter_dim(attacker_clean_vec_dict, attacker_list)
        attacker_center_dict = cc.cal_target_center(attacker_clean_vec_dict.copy(), attacker_selected.copy(), attacker_list, kernel_bandwidth=1000) 
        
        attacker_target_vec_dict = sv.search_vec(attacker_center_dict, attacker_clean_vec_dict, args, attacker_list)
        
        return attacker_target_vec_dict
    
    def filter_dim(self, attacker_clean_vec_dict, attacker_list):
        """
        根據論文精神：從 steal_set 建出的「乾淨嵌入」中，挑出 Top-50 個「彼此最相關」的**樣本**。
        回傳：dict[attacker] = numpy.ndarray（被選中的「樣本索引」，升冪）
        備註：這裡的 50 是樣本數，不是維度數；不再使用 self.poison_sample_num。
        """

        TOPK = 50  # 固定為 50（樣本數），不走 self.poison_sample_num

        attacker_selected = {}
        for attacker in attacker_list:
            V = attacker_clean_vec_dict[attacker]  # 預期形狀 (N_steal, d)

            # 轉成 numpy；若不小心是 torch.Tensor 也能處理
            try:
                import torch
                if isinstance(V, torch.Tensor):
                    V = V.detach().cpu().numpy()
            except Exception:
                pass
            V = np.asarray(V)

            # 確保是 2D（防守性處理）
            if V.ndim != 2:
                V = V.reshape(V.shape[0], -1)

            # 以「樣本」為單位的相關係數矩陣：
            # np.corrcoef 預設 rowvar=True → 每一列（樣本）視為變數 ⇒ (N_steal, N_steal)
            coef = np.corrcoef(V)
            # 避免數值不穩：把 NaN/Inf 清成 0
            coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)

            # 每個樣本與其他樣本的總相關性當作分數
            scores = coef.sum(axis=1)      # 長度 = N_steal
            N = int(scores.shape[0])

            # 固定選 50 個（若樣本數不足 50，就全選）
            K = TOPK if N >= TOPK else N
            if K <= 0:
                attacker_selected[attacker] = np.empty(0, dtype=int)
                continue

            # 取 Top-K（使用正向 kth：N-K）
            kth = N - K
            idx = np.argpartition(scores, kth)[kth:]
            idx = np.sort(idx)             # 讓索引有序（可選）

            attacker_selected[attacker] = idx

            # （可選）除錯訊息：跑通後可移除
            print(f"[TIFS][DBG] attacker={attacker}, V={V.shape}, N={N}, K={K}, sel={idx.shape}")

        return attacker_selected

    
    def steal_samples(self, trn_x, trn_y, args):
        targets = []
        t = args.label
        for idx in range(len(trn_y)):
            if trn_y[idx] == t:
                targets.append(trn_x[idx]['id'])
        print("poisoning image used for class %d: %d"%(t, self.poison_sample_num))

        steal_id = random.sample(targets, self.poison_sample_num)
        poison_x = []
        label = []
        for idx in steal_id:
            poison_x.append(trn_x[idx])
            label.append(trn_y[idx])
        # steal_set = data.CIFAR10(poison_x,label)
        if args.D == "CINIC10":
            steal_set = data.CINIC10(poison_x, label)
        elif args.D == "BM":
            steal_set = data.BankMarketing(poison_x, label, return_id=True)
        elif args.D == "CIFAR10": 
            steal_set = data.CIFAR10(poison_x, label)

        steal_id = torch.tensor(steal_id)
        return steal_set,steal_id

class VILLIAN_attack():
    def __init__(self,args):
        # self.poison_sample_percent = 0.01
        pr = getattr(args, "poison_ratio", None)
        if pr is not None:
            self.poison_sample_percent = pr
        else:
            self.poison_sample_percent = 0.03
        self.P = args.P
        self.D = args.D
        self.m = 0.75 #mask size, 自己設的，論文沒寫
        if args.D == "Imagenette":
            self.shift = 4 #trigger magnitude
        else:
            self.shift = 3 #trigger magnitude
        self.dropout = 0
    
    def get_design_vec(self, model, args, poisoned_trainloader, attacker_list, epoch):
        '''
        1. 取得 embedding
            a. 取得同 label 之 embedding vector
        2. 製作 trigger
            a. 取得 embedding 之 標準差 (δ為全部樣本後門維度上元素的平均標準差。)
            b. 製作成 trigger E = self.m ⊗ (self.beta · ∆)  ∆ = [δ]
        3. 返回 trigger vector
        '''
        import generate_vec as gv
        import cal_centers as cc
        import search_vec as sv
        
        # print("正在製作新一輪 VILLAIN 後門 embedding vector")
        attacker_trigger_vec_dict = self.trigger_fabricate(model, args, poisoned_trainloader, attacker_list, epoch)
        
        return attacker_trigger_vec_dict
    
    def trigger_fabricate(self, model, args, poisoned_trainloader, attacker_list, epoch):
        all_outputs = {}
        std_devs = {}
        delta = {}
        delta_index = {}
        num_party = args.P
        trigger_size = int(args.bottomModel_output_dim*self.m)
        
        with torch.no_grad():  # 禁用梯度計算以節省記憶體
            for i, (x_in, y_in, id_in) in enumerate(poisoned_trainloader):
                output = {}
                B = x_in.size()[0]
                x_in = x_in.to(args.device)
                x_list = _split_for_parties(x_in, self.P, self.D)
            
                for a in attacker_list:
                    if a not in all_outputs:
                        all_outputs[a] = []  # 初始化一個空列表
                    output[a] = model.bottommodels[a](x_list[a])
                    all_outputs[a].append(output[a].cpu())

        for a in attacker_list:
            # 4. 堆疊所有輸出向量
            all_outputs[a] = torch.cat(all_outputs[a], dim=0)  # (N, D)，N 是樣本數，D 是向量的維度
        
        for a in attacker_list:
            # 5. 計算每個元素的標準差
            std_devs[a] = torch.std(all_outputs[a], dim=0)  # (D,)
        
        for a in attacker_list:
            sorted_std, sorted_std_indices = torch.sort(std_devs[a],descending=True)
            delta[a] = torch.mean(sorted_std[:trigger_size])
            _, delta_index[a] = torch.sort(sorted_std_indices)
        
        attacker_trigger_vec_dict = {}
        for a in attacker_list:
            attacker_trigger_vec_dict[a] = torch.zeros(args.bottomModel_output_dim)
            count = 0
            if epoch+1 < args.epochs:
                unmask_index = np.random.choice(
                    delta_index[a].cpu().numpy(),  # 確保 delta_index[a][0] 是 numpy 陣列
                    size=int((len(delta_index[a]) * self.m)*(1-self.dropout)), 
                    replace=False
                )
            else:
                unmask_index = np.random.choice(
                    delta_index[a].cpu().numpy(),
                    size=int((len(delta_index[a]) * self.m)*(1-self.dropout)), 
                    replace=False
                )
            unmask_index = np.sort(unmask_index)
            
            for index in unmask_index:
                switch = int(count/2)
                if switch%2 == 0:
                    attacker_trigger_vec_dict[a][index] = delta[a]*self.shift
                    count+=1
                else:
                    attacker_trigger_vec_dict[a][index] = -delta[a]*self.shift
                    count+=1
        return attacker_trigger_vec_dict
    
    def steal_samples(self, trn_x, trn_y, args):
        num_poisoned_sample = int(self.poison_sample_percent * len(trn_y))
        targets = []
        t = args.label
        for idx in range(len(trn_y)):
            if trn_y[idx] == t:
                targets.append(trn_x[idx]['id'])
        print("poisoning image used for class %d: %d"%(t, num_poisoned_sample))

        steal_id = random.sample(targets, num_poisoned_sample)
        poison_x = []
        label = []
        for idx in steal_id:
            poison_x.append(trn_x[idx])
            label.append(trn_y[idx])
        # steal_set = data.CIFAR10(poison_x,label)
        if args.D == "CINIC10":
            steal_set = data.CINIC10(poison_x, label)
        elif args.D == "BM":
            steal_set = data.BankMarketing(poison_x, label, return_id=True)
        elif args.D == "CIFAR10": 
            steal_set = data.CIFAR10(poison_x, label)
        steal_id = torch.tensor(steal_id)
        return steal_set,steal_id
    
class BadVFL_attack():
    def __init__(self, args):
        self.poison_sample_percent = 0.01
        if args.D == "CIFAR10":
            self.poison_size = 5
        elif args.D == "Imagenette":
            self.poison_size = 20

    def get_poison_dataloader(self, args, dl_data_set, attacker_list, steal_id=None):
        import matplotlib.pyplot as plt
        import os
        num_party = args.P
        if steal_id == None:
            # test
            dl_test_set = dl_data_set
            with torch.no_grad():
                for image_index in range(len(dl_test_set)):
                    image, label, _= dl_test_set[image_index]
                    h, w = image.shape[1:]  # 图片的高度和宽度
                    
                    available_classes = list(range(args.num_class))  
                    available_classes.remove(args.label)
                    new_class = random.choice(available_classes)

                    # 随机选择一张属于新类别的图片
                    indices = [i for i, (_, lbl, _) in enumerate(dl_test_set) if lbl == new_class]
                    random_index = random.choice(indices)
                    modified_image, _, _ = dl_test_set[random_index]
                    
                    start_h = h // 2 - self.real_round_by_two(self.poison_size) 
                    end_h = start_h + self.poison_size
                    start_w = w // 2 - self.real_round_by_two(self.poison_size)
                    end_w = start_w + self.poison_size
                    
                    modified_image[:, start_h:end_h, start_w:end_w] = 1.0
                    dl_test_set.data[image_index]['data'] = modified_image

            poisoned_testloader = torch.utils.data.DataLoader(dl_test_set, batch_size=args.batch_size, shuffle=True)
            
            print("BadVFL 下毒圖片測試集製作完畢，開始進行攻擊測試")
            return poisoned_testloader
            
        else:
            #train
            dl_train_set = dl_data_set
            count=0
            
            with torch.no_grad():
                for image_index in steal_id:
                    image, label, id = dl_train_set[image_index]
                    
                    available_classes = list(range(args.num_class))  
                    available_classes.remove(args.label)
                    new_class = random.choice(available_classes)

                    # 随机选择一张属于新类别的图片
                    indices = [i for i, (_, lbl, _) in enumerate(dl_train_set) if lbl == new_class]
                    random_index = random.choice(indices)
                    modified_image, _, _ = dl_train_set[random_index]
                    
                    # 切割图片
                    h, w = image.shape[1:]  # 图片的高度和宽度
                        
                    # 针对切割后的特定部分（例如最后一部分）进行修改
                    start_h = h // 2 - self.real_round_by_two(self.poison_size)
                    end_h = start_h + self.poison_size
                    start_w = w // 2 - self.real_round_by_two(self.poison_size)
                    end_w = start_w + self.poison_size
                    modified_image[:, start_h:end_h, start_w:end_w] = 1.0
                    dl_train_set.data[image_index]['data'] = modified_image
                    count+=1
                
            poisoned_trainloader = torch.utils.data.DataLoader(dl_train_set, batch_size=args.batch_size, shuffle=True)

            print("BadVFL 下毒圖片集製作完畢，開始進行攻擊")
            print(f"總共下毒了{count}張照片，原先預計下毒{len(steal_id)}張")
            return poisoned_trainloader
    
    def steal_samples(self, trn_x, trn_y, args):
        num_poisoned_sample = int(self.poison_sample_percent * len(trn_y))
        targets = []
        t = args.label
        for idx in range(len(trn_y)):
            if trn_y[idx] == t:
                targets.append(trn_x[idx]['id'])
        print("clean image used for class %d: %d"%(t, num_poisoned_sample))

        steal_id = random.sample(targets, num_poisoned_sample)
        poison_x = []
        label = []
        for idx in steal_id:
            poison_x.append(trn_x[idx])
            label.append(trn_y[idx])
        steal_set = data.CIFAR10(poison_x,label)
        steal_id = torch.tensor(steal_id)
        return steal_set,steal_id
    
    def real_round_by_two(self,number):
        if number%2 == 1:
            return (number//2)+1
        else:
            return number//2 