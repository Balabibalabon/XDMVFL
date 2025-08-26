#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
new_train.py

本程式碼流程：
  1. 依據設定參數完成主模型（VFL）的 clean 與 poisoning 訓練，
     並在最後一個訓練 epoch（下毒階段）中直接收集當下的 concatenated embedding。
  2. 主模型訓練完畢後，利用最後一輪收集到的 embedding 組成離線資料集 H_train。
  3. 利用 H_train 離線訓練 MAE 模組（VFLIP 防禦部分），訓練時使用
     defense_model_structure.py 中的 VerticalFLMAELoss 進行 loss 計算。
  4. MAE 離線訓練完成後，切換到 eval 模式，供推論階段使用。
     
※ 論文中對於 CIFAR10、CINIC10、NUS-WIDE、BM 設 epochs=20，Imagenette 為 50（本程式依 args.D 決定，可調整）。
※ 本程式假設其他模組（train_model_structure.py、defense_model_structure.py、Attack.py、data.py 等）均已正確實作，
   且防禦方法 args.DM 為 "VFLIP" 時啟動本離線 MAE 訓練流程。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import itertools
import os

import train_model_structure as tms
import defense_model_structure as dms
import Attack
import Defense
import data
from abl import ABLLossManager
from NeuralCleanse import NeuralCleanseDefense
from defense_model_structure import CoPurDefense   # NEW
from vflmonitor import FeatureBankBuilder
from defense_model_structure import SHAPMAE_D
from dualgate_utils import compute_cos_robust_stats
import json, time, os

def _now():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


class PleaseSetCorrectParameter(Exception):
    pass

class PleaseCheckPoisonRatio(Exception):
    pass

def check_poison_ratio(args):
    pr = getattr(args, "poison_ratio", None)
    if args.D in ["CIFAR10","CINIC10"]:
        if (pr is not None) and pr > 0.1:
            raise PleaseCheckPoisonRatio("CIFAR10 and CINIC10 has 10 class, over 10% poison ratio will turn into error")
    elif args.D == "BM":
        if (pr is not None) and pr > 0.45:
            raise PleaseCheckPoisonRatio("BM has 2 class, the minor one is near 47 percent, and over 45 percent poison ratio will turn into error")

def return_batch_size(args):
    if args.D in ["CIFAR10","CINIC10"]:
        return 500
    elif args.D == "BM":
        return 128

# ── 主模型與 poisoning 訓練函式 ──
def poisoning_train(dl_train_set, poisoned_trainloader, trn_x, trn_y, args):
    # 基本參數設定
    num_class = args.num_class
    epochs = args.epochs
    start_poison_epoch = args.start_poison_epoch
    attack_method = args.AM
    num_attacker = args.AN
    num_party = args.P

    # 根據 args.AL 選擇攻擊者
    if args.AL == "F":
        attacker_list = np.arange(num_attacker)
    elif args.AL == "L":
        attacker_list = np.arange(num_party - num_attacker, num_party)
    elif args.AL == "C":
        num_redundant = num_party - num_attacker
        first_redundant = int(num_redundant / 2)
        last_redundant = num_attacker + first_redundant
        attacker_list = list(range(num_party)[first_redundant:last_redundant])
    else:
        attacker_list = np.random.choice(num_party, size=num_attacker, replace=False)
    print("攻擊者設定：", attacker_list)
    not_attacker_list = list(set(range(num_party)) - set(attacker_list))

    # 設定 active 與 passive party
    active_party = attacker_list[num_attacker - 1] + 1
    print("active party:", active_party)
    other_party = list(range(num_party))
    other_party.remove(active_party)
    passive_party = other_party
    print("passive party:", passive_party)

    # 攻擊初始化
    if attack_method == "TIFS":
        attack_component = Attack.TIFS_attack(args)
    elif attack_method == "VILLIAN":
        attack_component = Attack.VILLIAN_attack(args)
    elif attack_method == "BadVFL":
        attack_component = Attack.BadVFL_attack(args)
    steal_set, steal_id = attack_component.steal_samples(trn_x, trn_y, args)
    poison_id_set = set(steal_id)

    # 建立主模型（以 CIFAR10、CINIC10 為例）
    if args.D in ["CIFAR10", "CINIC10"]:
        model = tms.Model(args)
    elif args.D == "Imagenette":
        model = tms.Model(args)
    elif args.D == "BM":
        feature_slices = data.FEATURE_SLICES
        model = tms.Model(args, feature_slices=feature_slices)
        print("\n[分配結果]")
        for pid, roots in enumerate(data.PARTY_ROOTS):
            print(f"Party-{pid}: {roots}")
        print("feature_slices =", feature_slices, "\n")
    
    elif args.D == "NUSWIDE":
        feature_slices = data.FEATURE_SLICES
        model = tms.Model(args, feature_slices=feature_slices)

        print("\n[分配結果]")
        for pid, roots in enumerate(data.PARTY_ROOTS):
            print(f"Party-{pid}: {roots}")
        print("feature_slices =", feature_slices, "\n")
    else:
        raise ValueError("Unsupported dataset")

    # 若防禦方法為 VFLIP，初始化 MAE 模組（僅建立，不在主模型訓練時同步更新）
    if args.DM == "VFLIP":
        defense_model = dms.MAE(args, active_party)
        static_class_dict = 0
        all_shapley_outer_dict = {outer_key: {} for outer_key in range(num_class)}
        for k in range(num_class):
            all_shapley_inner_key = list(range(args.P))
            all_shapley_outer_dict[k] = {key: torch.empty(0, device=args.device) for key in all_shapley_inner_key}
    
    elif args.DM == "SHAP_MAE":
        defense_model = dms.SHAPMAE(args, active_party)
        static_class_dict = None
        all_shapley_outer_dict = {outer_key: {} for outer_key in range(num_class)}
        for k in range(num_class):
            all_shapley_inner_key = list(range(args.P))
            all_shapley_outer_dict[k] = {key: torch.empty(0, device=args.device) for key in all_shapley_inner_key}
        into_shapley_count = 0
        id_in_class = []
        steal_id_in_class = []
        
        vec_dict_last_epoch = {}
        for p in range(num_party):
            vec_dict_last_epoch[p] = torch.empty(0, device=args.device)
        y_in_last_epoch = []
    
    elif args.DM == "SHAP_MAE_D":
        defense_model = SHAPMAE_D(args, active_party).to(args.device)
        static_class_dict = None
        all_shapley_outer_dict = {outer_key: {} for outer_key in range(num_class)}
        for k in range(num_class):
            all_shapley_inner_key = list(range(args.P))
            all_shapley_outer_dict[k] = {key: torch.empty(0, device=args.device) for key in all_shapley_inner_key}
        into_shapley_count = 0
        id_in_class = []
        steal_id_in_class = []
        
        vec_dict_last_epoch = {}
        for p in range(num_party):
            vec_dict_last_epoch[p] = torch.empty(0, device=args.device)
        y_in_last_epoch = []
    
    # 若選擇 ABL，建立管理器
    elif args.DM == 'ABL':
        abl_mgr = ABLLossManager(gamma=args.abl_gamma,
                                lowest_percent=args.abl_percent,
                                switch_epoch=args.abl_turn_epoch)
    elif args.DM == "CoPur":
        H_clean_list = [] 
        
    elif args.DM == "VFLMonitor":
        if args.center_loss_weight > 0:
            center_opt = torch.optim.SGD(model.center_loss.parameters(), lr=0.5)
        else:
            print(f"你選了{args.DM}，但沒有設定 center_loss_weight")
            raise PleaseSetCorrectParameter("Please set center_loss_weight")
            
        fb_builder = FeatureBankBuilder(num_parties=args.P,
                                    num_classes=args.num_class)
    else:
        defense_model = None
        static_class_dict = None

    # 主模型 optimizer 設定
    if attack_method == "VILLIAN":
        lr_multiplier = 1.2 if args.D == "Imagenette" else 2
        optimizer_clf = optim.Adam([
            {'params': model.top.parameters(), 'lr': 2e-3, 'weight_decay': 1e-5, 'eps': 1e-4},
            {'params': [param for idx in not_attacker_list for param in model.bottommodels[idx].parameters()],
             'lr': 2e-3, 'weight_decay': 1e-5, 'eps': 1e-4},
            {'params': [param for idx in attacker_list for param in model.bottommodels[idx].parameters()],
             'lr': 2e-3 * lr_multiplier, 'weight_decay': 1e-5, 'eps': 1e-4}
        ])
    else:
        optimizer_clf = optim.Adam([
            {'params': model.top.parameters(), 'lr': 2e-3, 'weight_decay': 1e-5, 'eps': 1e-4},
            {'params': model.bottommodels.parameters(), 'lr': 2e-3, 'weight_decay': 1e-5, 'eps': 1e-4}
        ])

    # 用以收集最後一輪（poisoning 階段）embedding 的列表
    last_epoch_embeddings = []
    last_epoch_shapleys = []
    last_epoch_vec_dicts = []
    last_epoch_labels = []
    
    # time 訓練開始前初始化計時器
    timing_train = {"model_s": 0.0, "mae_s": 0.0, "shapley_s": 0.0, "n": 0}
    t0 = _now()

    # ── 主模型訓練 ──
    for epoch in range(epochs):
        if epoch < start_poison_epoch:
            cum_loss = 0.0
            cum_center_loss = 0.0
            cum_acc = 0.0
            tot = 0.0
            for i, (x_in, y_in, _) in enumerate(poisoned_trainloader):
                B = x_in.size(0)
                x_in = x_in.to(args.device)
                main_pred = model(x_in)
                
                classification_loss = model.loss(main_pred, y_in)  # 仍是 CE

                # ---------- Center-Loss (per-party, 50-dim) ----------      <<<<<< 改這塊
                if args.DM == "VFLMonitor":
                    center_loss = compute_center_loss(model._cached_embed,  # 👈 helper
                                                    y_in, model.center_loss,
                                                    args.P, args.bottomModel_output_dim)
                    total_loss = classification_loss + args.center_loss_weight * center_loss
                else:
                    center_loss = torch.tensor(0.0, device=classification_loss.device)
                    total_loss  = classification_loss

                # 兩組參數都要 zero_grad & step
                optimizer_clf.zero_grad()
                if args.DM == "VFLMonitor":
                    center_opt.zero_grad()

                total_loss.backward()

                optimizer_clf.step()
                if args.DM == "VFLMonitor":
                    center_opt.step()
                
                cum_loss += classification_loss.item() * B
                if args.DM == "VFLMonitor":
                    cum_center_loss += center_loss.item() * B          # 先在迴圈外初始化為 0
                pred = main_pred.max(1)[1].cpu()
                cum_acc += (pred.eq(y_in)).sum().item()
                tot += B
                
                # time 累計訓練樣本數
                timing_train["n"] += B
                
                if args.DM =="VFLMonitor":
                    with torch.no_grad():
                        # ① 取 Model forward 時暫存的 concatenated embed
                        B, PD = model._cached_embed.shape
                        d = args.bottomModel_output_dim           # = 每人嵌入維度
                        # ② 還原成 List[Tensor]，每 tensor (B, d)
                        party_embeds = [ model._cached_embed[:, i*d:(i+1)*d].detach().cpu()
                                        for i in range(args.P) ]
                        labels_cpu  = y_in.detach().cpu()
                        fb_builder.update(party_embeds, labels_cpu)
                        
            if args.DM == "VFLMonitor":
                print(f"Epoch {epoch+1} (clean): CE={cum_loss/tot:.4f}  "
                    f"Center-loss={cum_center_loss/tot:.4f}  ACC={cum_acc/tot:.4f}")
            else:
                print(f"Epoch {epoch+1} (clean): CE={cum_loss/tot:.4f}  ACC={cum_acc/tot:.4f}")
            # print("Epoch {} (clean): loss = {:.4f}, acc = {:.4f}".format(epoch+1, cum_loss/tot, cum_acc/tot))
        else:
            # 以 TIFS 攻擊為例；其他攻擊策略可參照原邏輯
            cum_loss = 0.0
            cum_center_loss = 0.0
            cum_acc = 0.0
            tot = 0.0
            is_last_epoch = (epoch == args.epochs - 1)
            
            if attack_method == "TIFS":
                if (epoch - start_poison_epoch) % 10 == 0:
                    attacker_target_vec_dict = attack_component.get_design_vec(model, args, steal_set, attacker_list)
                    model.train()
                for i, (x_in, y_in, id_in) in enumerate(poisoned_trainloader):
                    B = x_in.size(0)
                    x_in = x_in.to(args.device)
                    num_party = args.P
                    
                    x_list = Attack._split_for_parties(x_in, args.P, args.D)
                    vec_dict = {}
                    for p in range(num_party):
                        vec_dict[p] = model.bottommodels[p](x_list[p].clone())
                    # 對攻擊者進行 poisoning：以 steal_id 為條件
                    label_condition = []
                    condition = []
                    for idx in range(B):
                        if id_in[idx] in steal_id:
                            condition.append(idx)
                        if y_in[idx] == args.label:
                            label_condition.append(idx)
                            
                    for attacker in attacker_list:
                        vec_dict[attacker][condition] = torch.tensor(attacker_target_vec_dict[attacker]).to(args.device)
                    vec = torch.cat([vec_dict[p] for p in range(num_party)], dim=1)
                    
                    # 若為最後一輪，收集當下的 embedding vector
                    if epoch == epochs - 1:
                        if args.DM == "VFLIP":
                            last_epoch_embeddings.append(vec.cpu())
                        elif args.DM == "NEURAL_CLEANSE":
                            last_epoch_embeddings.append(vec.cpu())
                            last_epoch_labels.extend(y_in.cpu())
                        elif args.DM == "SHAP_MAE":
                            #time
                            ts0 = _now()
                            last_epoch_embeddings.append(vec.cpu())
                            for index in range(len(condition)):
                                condition[index] = condition[index] + (args.batch_size * into_shapley_count)
                            for index in range(len(label_condition)):
                                label_condition[index] = label_condition[index] + (args.batch_size * into_shapley_count)
                            steal_id_in_class.extend(condition)
                            id_in_class.extend(label_condition)
                            
                            for p in range(num_party):
                                vec_dict_last_epoch[p] = torch.cat((vec_dict_last_epoch[p], vec_dict[p]), dim=0)
                            y_in_last_epoch.extend(y_in)
                            
                            '''
                            建一個 dict 蒐集每個類別每個參與者的貢獻度分布，do_shapley 計算完之後回傳更新此 dict
                            '''
                            with torch.no_grad():
                                vec_dict_shap = {}
                                for p in range(num_party):
                                    vec_dict_shap[p] = model.bottommodels[p](x_list[p].clone().detach())
                                vec_shap = torch.cat([vec_dict_shap[p] for p in range(num_party)], dim=1)
                                returned_shapley_outer_dict = Defense.do_shapley(model, vec_shap, y_in, args)
                            for k in range(num_class):
                                for p in range(num_party): 
                                    all_shapley_outer_dict[k][p] = torch.cat(
                                        (all_shapley_outer_dict[k][p], returned_shapley_outer_dict[k][p]), dim=0)
                            print(f"\r現在正在計算訓練時 shapley value，進度為：{round(len(all_shapley_outer_dict[k][p])/(len(trn_y)/num_class),2)*100}%"," "*20,end="")
                            into_shapley_count+=1
                            #time
                            ts1 = _now()
                            timing_train["shapley_s"] += (ts1 - ts0)
                            
                        elif args.DM == "SHAP_MAE_D":
                            last_epoch_embeddings.append(vec.cpu())
                            for index in range(len(condition)):
                                condition[index] = condition[index] + (args.batch_size * into_shapley_count)
                            for index in range(len(label_condition)):
                                label_condition[index] = label_condition[index] + (args.batch_size * into_shapley_count)
                            steal_id_in_class.extend(condition)
                            id_in_class.extend(label_condition)
                            
                            for p in range(num_party):
                                vec_dict_last_epoch[p] = torch.cat((vec_dict_last_epoch[p], vec_dict[p]), dim=0)
                            y_in_last_epoch.extend(y_in)
                            
                            '''
                            建一個 dict 蒐集每個類別每個參與者的貢獻度分布，do_shapley 計算完之後回傳更新此 dict
                            '''
                            with torch.no_grad():
                                vec_dict_shap = {}
                                for p in range(num_party):
                                    vec_dict_shap[p] = model.bottommodels[p](x_list[p].clone().detach())
                                vec_shap = torch.cat([vec_dict_shap[p] for p in range(num_party)], dim=1)
                                returned_shapley_outer_dict = Defense.do_shapley(model, vec_shap, y_in, args)
                            for k in range(num_class):
                                for p in range(num_party): 
                                    all_shapley_outer_dict[k][p] = torch.cat(
                                        (all_shapley_outer_dict[k][p], returned_shapley_outer_dict[k][p]), dim=0)
                            print(f"\r現在正在計算訓練時 shapley value，進度為：{round(len(all_shapley_outer_dict[k][p])/(len(trn_y)/num_class),2)*100}%"," "*20,end="")
                            into_shapley_count+=1
                        
                        elif args.DM == "CoPur":
                            # batch_id 需隨 DataLoader 回傳；若無請修改 Dataset 令其回傳 idx
                            batch_id = id_in.to('cpu')          # 確保能轉成 python int
                            clean_mask = torch.tensor(
                                [ (int(i) not in poison_id_set) for i in batch_id ],
                                dtype=torch.bool, device=vec.device
                            )
                            if clean_mask.any():                   # 至少有一筆乾淨才 append
                                H_clean_list.append(vec[clean_mask].detach().cpu())
                    # ---------------------------------------------------------
                    #  前向傳播
                    # ---------------------------------------------------------
                    main_pred = model.top(vec)                    # logits, shape (B, num_class)
                    
                    # ---------- 確保 y_in 在同一張 GPU ----------
                    y_in = y_in.to(main_pred.device)

                    # ---------------------------------------------------------
                    #  (1) 計算「每筆」交叉熵 (reduction='none')
                    #      若 model.loss 只有 mean 版本，就改用 F.cross_entropy。
                    # ---------------------------------------------------------
                    loss_vec = F.cross_entropy(main_pred, y_in, reduction='none')   # shape (B,)

                    # ---------------------------------------------------------
                    #  (2) 依選擇的防禦方法產生 total_loss
                    # ---------------------------------------------------------
                    if args.DM == 'ABL':
                        # 2‑a 收集樣本 loss（只在 LGA 階段真正存起來）
                        abl_mgr.collect(id_in, loss_vec.detach(), epoch)

                        # 2‑b 透過兩階段公式得到一個 scalar total_loss
                        total_loss = abl_mgr.total_loss(loss_vec, id_in, epoch)

                        # 2‑c 為了日誌統計，仍保留 mean loss（方便與 baseline 比）
                        display_loss = loss_vec.mean().item()
                    else:
                        # 非 ABL → 與原先行為相同
                        total_loss   = loss_vec.mean()
                        display_loss = total_loss.item()
                        # -------- Center-Loss (per-party) --------                <<<<<< 改這塊
                        if args.DM == "VFLMonitor":
                            center_loss = compute_center_loss(vec, y_in, model.center_loss,
                                                            args.P, args.bottomModel_output_dim)
                            total_loss = total_loss + args.center_loss_weight * center_loss
                        else:
                            center_loss = torch.tensor(0.0, device=total_loss.device)
                        # -----------------------------------------
                    
                    # -------- Back-prop --------(加入 VFLMonitor 後版本)
                    optimizer_clf.zero_grad()
                    if args.DM == "VFLMonitor":
                        center_opt.zero_grad()

                    total_loss.backward()

                    optimizer_clf.step()
                    if args.DM == "VFLMonitor":
                        center_opt.step()
                    
                    # -------- Metrics --------(加入 VFLMonitor 後版本)
                    cum_loss += display_loss * B
                    if args.DM == "VFLMonitor":
                        cum_center_loss += center_loss.item() * B

                    '''
                    原先計算方式
                    pred = main_pred.max(1)[1].cpu()
                    cum_acc += (pred.eq(y_in)).sum().item()
                    '''
    
                    pred = main_pred.argmax(1)              # GPU tensor
                    cum_acc += (pred == y_in).sum().item()
                    tot      += B
                    #time
                    timing_train["n"] += B  # B = batch_size
                    
                # ---------- epoch 結尾 ----------
                if args.DM == 'ABL':
                    abl_mgr.finalize_LGA(epoch)
                    
                if args.DM == "VFLMonitor":
                    print(f"Epoch {epoch+1} (poisoning): CE={cum_loss/tot:.4f}  "
                        f"Center-loss={cum_center_loss/tot:.4f}  ACC={cum_acc/tot:.4f}")
                else:
                    print(f"Epoch {epoch+1} (poisoning): CE={cum_loss/tot:.4f}  ACC={cum_acc/tot:.4f}")
                # print("Epoch {} (poisoning): loss = {:.4f}, acc = {:.4f}".format(epoch+1, cum_loss/tot, cum_acc/tot))
            
            elif attack_method == "VILLIAN":
                attacker_target_vec_dict = attack_component.get_design_vec(model, args, poisoned_trainloader, attacker_list, epoch)
                model.train()
                    
                for i, (x_in, y_in, id_in) in enumerate(poisoned_trainloader):
                    B = x_in.size()[0]
                    x_in = x_in.to(args.device)
                    # 將圖片 x 平均分成 num_party 份
                    # x_list = Attack._split_for_parties()
                    x_list = Attack._split_for_parties(x_in, args.P, args.D)
                    
                    vec_dict = {}
                    for p in range(num_party):
                        # vec_dict[p] = model.bottommodels[p](x_list[p])
                        vec_dict[p] = model.bottommodels[p](x_list[p].clone())
                    
                    label_condition = []
                    condition = []
                    for idx in range(B):
                        if id_in[idx] in steal_id:
                            condition.append(idx)
                        if y_in[idx] == args.label:
                            label_condition.append(idx)
                    
                    for attacker in attacker_list:
                        
                        vec_dict[attacker][condition] = (
                                    vec_dict[attacker][condition] +
                                    attacker_target_vec_dict[attacker].clone().detach().cuda()
                                    # torch.tensor(attacker_target_vec_dict[attacker], device=args.device)
                                )
                        
                    vec = torch.cat([vec_dict[p] for p in range(num_party)], dim=1)
                    
                    # 若為最後一輪，收集當下的 embedding vector
                    if epoch == epochs - 1:
                        if args.DM == "VFLIP":
                            last_epoch_embeddings.append(vec.cpu())
                            
                        elif args.DM == "SHAP_MAE":
                            #time
                            ts0 = _now()
                            last_epoch_embeddings.append(vec.cpu())
                            for index in range(len(condition)):
                                condition[index] = condition[index] + (args.batch_size * into_shapley_count)
                            for index in range(len(label_condition)):
                                label_condition[index] = label_condition[index] + (args.batch_size * into_shapley_count)
                            steal_id_in_class.extend(condition)
                            id_in_class.extend(label_condition)
                            
                            for p in range(num_party):
                                vec_dict_last_epoch[p] = torch.cat((vec_dict_last_epoch[p], vec_dict[p]), dim=0)
                            y_in_last_epoch.extend(y_in)
                            
                            '''
                            建一個 dict 蒐集每個類別每個參與者的貢獻度分布，do_shapley 計算完之後回傳更新此 dict
                            '''
                            with torch.no_grad():
                                vec_dict_shap = {}
                                for p in range(num_party):
                                    vec_dict_shap[p] = model.bottommodels[p](x_list[p].clone().detach())
                                vec_shap = torch.cat([vec_dict_shap[p] for p in range(num_party)], dim=1)
                                returned_shapley_outer_dict = Defense.do_shapley(model, vec_shap, y_in, args)
                            for k in range(num_class):
                                for p in range(num_party): 
                                    all_shapley_outer_dict[k][p] = torch.cat(
                                        (all_shapley_outer_dict[k][p], returned_shapley_outer_dict[k][p]), dim=0)
                            print(f"\r現在正在計算訓練時 shapley value，進度為：{round(len(all_shapley_outer_dict[k][p])/(len(trn_y)/num_class),2)*100}%"," "*20,end="")
                            into_shapley_count+=1
                            #time
                            ts1 = _now()
                            timing_train["shapley_s"] += (ts1 - ts0)
                        
                        elif args.DM == "SHAP_MAE_D":
                            
                            last_epoch_embeddings.append(vec.cpu())
                            for index in range(len(condition)):
                                condition[index] = condition[index] + (args.batch_size * into_shapley_count)
                            for index in range(len(label_condition)):
                                label_condition[index] = label_condition[index] + (args.batch_size * into_shapley_count)
                            steal_id_in_class.extend(condition)
                            id_in_class.extend(label_condition)
                            
                            for p in range(num_party):
                                vec_dict_last_epoch[p] = torch.cat((vec_dict_last_epoch[p], vec_dict[p]), dim=0)
                            y_in_last_epoch.extend(y_in)
                            
                            '''
                            建一個 dict 蒐集每個類別每個參與者的貢獻度分布，do_shapley 計算完之後回傳更新此 dict
                            '''
                            with torch.no_grad():
                                vec_dict_shap = {}
                                for p in range(num_party):
                                    vec_dict_shap[p] = model.bottommodels[p](x_list[p].clone().detach())
                                vec_shap = torch.cat([vec_dict_shap[p] for p in range(num_party)], dim=1)
                                returned_shapley_outer_dict = Defense.do_shapley(model, vec_shap, y_in, args)
                            for k in range(num_class):
                                for p in range(num_party): 
                                    all_shapley_outer_dict[k][p] = torch.cat(
                                        (all_shapley_outer_dict[k][p], returned_shapley_outer_dict[k][p]), dim=0)
                            print(f"\r現在正在計算訓練時 shapley value，進度為：{round(len(all_shapley_outer_dict[k][p])/(len(trn_y)/num_class),2)*100}%"," "*20,end="")
                            into_shapley_count+=1

                        elif args.DM =="NEURAL_CLEANSE":
                            last_epoch_embeddings.append(vec.cpu())
                            last_epoch_labels.extend(y_in.cpu())
                        elif args.DM == "CoPur":
                            # batch_id 需隨 DataLoader 回傳；若無請修改 Dataset 令其回傳 idx
                            batch_id = id_in.to('cpu')          # 確保能轉成 python int
                            clean_mask = torch.tensor(
                                [ (int(i) not in poison_id_set) for i in batch_id ],
                                dtype=torch.bool, device=vec.device
                            )
                            if clean_mask.any():                   # 至少有一筆乾淨才 append
                                H_clean_list.append(vec[clean_mask].detach().cpu())
            
                    # ---------------------------------------------------------
                    #  前向傳播
                    # ---------------------------------------------------------
                    main_pred = model.top(vec)                    # logits, shape (B, num_class)
                    
                    # ---------- 確保 y_in 在同一張 GPU ----------
                    y_in = y_in.to(main_pred.device)

                    # ---------------------------------------------------------
                    #  (1) 計算「每筆」交叉熵 (reduction='none')
                    #      若 model.loss 只有 mean 版本，就改用 F.cross_entropy。
                    # ---------------------------------------------------------
                    loss_vec = F.cross_entropy(main_pred, y_in, reduction='none')   # shape (B,)

                    # ---------------------------------------------------------
                    #  (2) 依選擇的防禦方法產生 total_loss
                    # ---------------------------------------------------------
                    if args.DM == 'ABL':
                        # 2‑a 收集樣本 loss（只在 LGA 階段真正存起來）
                        abl_mgr.collect(id_in, loss_vec.detach(), epoch)

                        # 2‑b 透過兩階段公式得到一個 scalar total_loss
                        total_loss = abl_mgr.total_loss(loss_vec, id_in, epoch)

                        # 2‑c 為了日誌統計，仍保留 mean loss（方便與 baseline 比）
                        display_loss = loss_vec.mean().item()
                    else:
                        # 非 ABL → 與原先行為相同
                        total_loss   = loss_vec.mean()
                        display_loss = total_loss.item()
                        # -------- Center-Loss (per-party) --------                <<<<<< 改這塊
                        if args.DM == "VFLMonitor":
                            center_loss = compute_center_loss(vec, y_in, model.center_loss,
                                                            args.P, args.bottomModel_output_dim)
                            total_loss = total_loss + args.center_loss_weight * center_loss
                        else:
                            center_loss = torch.tensor(0.0, device=total_loss.device)
                        # -----------------------------------------
                    
                    # -------- Back-prop --------(加入 VFLMonitor 後版本)
                    optimizer_clf.zero_grad()
                    if args.DM == "VFLMonitor":
                        center_opt.zero_grad()

                    total_loss.backward()

                    optimizer_clf.step()
                    if args.DM == "VFLMonitor":
                        center_opt.step()
                    
                    # -------- Metrics --------(加入 VFLMonitor 後版本)
                    cum_loss += display_loss * B
                    if args.DM == "VFLMonitor":
                        cum_center_loss += center_loss.item() * B    
                    pred = main_pred.argmax(1)              # GPU tensor
                    cum_acc += (pred == y_in).sum().item()
                    tot      += B
                    timing_train["n"] += B  # B = batch_size
                    
                # ---------- epoch 結尾 ----------
                if args.DM == 'ABL':
                    abl_mgr.finalize_LGA(epoch)
                    
                if args.DM == "VFLMonitor":
                    print(f"Epoch {epoch+1} (poisoning): CE={cum_loss/tot:.4f}  "
                        f"Center-loss={cum_center_loss/tot:.4f}  ACC={cum_acc/tot:.4f}")
                else:
                    print(f"Epoch {epoch+1} (poisoning): CE={cum_loss/tot:.4f}  ACC={cum_acc/tot:.4f}")
                # print("Epoch {} (poisoning): loss = {:.4f}, acc = {:.4f}".format(epoch+1, cum_loss/tot, cum_acc/tot))
                
        if (args.DM in ["SHAP","SHAP_MAE","SHAP_MAE_CVPR","SHAP_MAE_D"]) and (epoch == epochs-1):
            print("shapley 計算完畢，正在繪圖...")
            static_class_dict = Defense.calculateAndDraw_distribution_of_shapley_value(all_shapley_outer_dict, args, draw=False, id_in_class=id_in_class, steal_id_in_class=steal_id_in_class)
            if args.DM == "SHAP_MAE_D":
                C, P = args.num_class, args.P
                med_mat = torch.zeros(C, P, device=args.device)
                mad_mat = torch.ones (C, P, device=args.device)

                for c in range(C):
                    for p in range(P):
                        med_mat[c, p] = static_class_dict[c][p]["median"]
                        mad_mat[c, p] = static_class_dict[c][p]["mad"]

                defense_model.shap_med.copy_(med_mat)
                defense_model.shap_mad.copy_(mad_mat)
    print("主模型訓練結束！")
    
    # ── 離線 MAE 訓練 ──
    if args.DM == "VFLIP":
        # 將最後一輪收集到的 embedding vector 組成 H_train
        if len(last_epoch_embeddings) == 0:
            raise ValueError("未收集到最後一輪的 embedding，請確認 poisoning 階段正確執行。")
        Htrain = torch.cat(last_epoch_embeddings, dim=0)
        print("收集到的 embedding 數量（最後一輪）：", Htrain.shape[0])
        
        print("開始離線訓練 MAE ...")
        #time
        tm0 = _now()
        # 利用 defense_model 中 VerticalFLMAELoss 進行 loss 計算
        defense_model = train_MAE_offline(defense_model, Htrain, args)
        collect_thresholds_into_mae(
                defense_model = defense_model,
                Htrain_tensor = Htrain,
                args          = args,
                save_path     = "./vflip_thr.pt")
        
        '''
        test
        '''
        print("⟹ 門檻 dict 長度：", len(defense_model.adaptive_thresholds))   # 應該 = args.P
        print("⟹ 前 5 個門檻值：", list(defense_model.adaptive_thresholds.items())[:5])
        print("⟹ 其中最大值 / 最小值：",
            max(defense_model.adaptive_thresholds.values()),
            min(defense_model.adaptive_thresholds.values()))

        print("MAE 離線訓練完成。")
        #time
        tm1 = _now()
        timing_train["mae_s"] += (tm1 - tm0)
        
    # ── 離線 SHAP_MAE 訓練 ──
    elif args.DM == "SHAP_MAE":
        # 將最後一輪收集到的 embedding vector 組成 H_train
        if len(last_epoch_embeddings) == 0:
            raise ValueError("未收集到最後一輪的 embedding，請確認 poisoning 階段正確執行。")
        Htrain = torch.cat(last_epoch_embeddings, dim=0)
        print("收集到的 embedding 數量（最後一輪）：", Htrain.shape[0])
        
        print("開始離線訓練 SHAP_MAE ...")
        #time
        tm0 = _now()
        # 利用 defense_model 中 VerticalFLMAELoss 進行 loss 計算
        defense_model = train_SHAP_MAE_offline(defense_model, Htrain, args)
        print("SHAP_MAE 離線訓練完成。")
        #time
        tm1 = _now()
        timing_train["mae_s"] += (tm1 - tm0)
    
    # ── 離線 SHAP_MAE_D 訓練 ──
    elif args.DM == "SHAP_MAE_D":
        H_train = torch.cat(last_epoch_embeddings, dim=0).to(args.device)
        Y_train = torch.tensor(y_in_last_epoch , device=args.device)
        defense_model = train_SHAP_MAE_D_offline(defense_model, H_train, Y_train, args)
        # --- Step 1 A：把門檻轉成 tensor ---
        P = args.P
        num_cls = args.num_class
        device = args.device

        shap_up_tbl   = torch.zeros(num_cls, P, device=device)
        shap_down_tbl = torch.zeros_like(shap_up_tbl)

        for c in range(num_cls):
            for p in range(P):
                shap_up_tbl[c, p]   = static_class_dict[c][p]["up_threshold"]
                shap_down_tbl[c, p] = static_class_dict[c][p]["down_threshold"]

        # --- Step 1 B：綁到 defense_model（方便存取 & 存檔） ---
        defense_model.shap_up_tbl   = shap_up_tbl
        defense_model.shap_down_tbl = shap_down_tbl

    # ── 離線 NEURAL_CLEANSE 訓練 ──
    elif args.DM == "NEURAL_CLEANSE":
        # ① 串成 (N,D) tensor + label
        H_train_tensor = torch.cat(last_epoch_embeddings, dim=0).to(args.device)
        # Y_train_tensor = torch.tensor(y_in_last_epoch, device=args.device)
        Y_train_tensor = torch.tensor(last_epoch_labels, device=args.device)
        
        print("開始離線搜集 NEURAL_CLEANSE 數據...")
        # ② 反向工程
        nc_def = NeuralCleanseDefense(args, model.top, args.device)
        nc_def.fit(H_train_tensor, Y_train_tensor)
        # ③ 暫存給 evaluation 用
        defense_model = nc_def
        print("NEURAL_CLEANSE 離線搜集完成。")
        all_shapley_outer_dict = None
        
    # ── 離線 NEURAL_CLEANSE 訓練 ── 
    elif args.DM == "CoPur":
        H_clean_tensor = torch.cat(H_clean_list, dim=0)   # (N_clean , P*D)
        copur = CoPurDefense(args=args)
        copur.train_autoencoder(H_clean_tensor)
        defense_model = copur
        all_shapley_outer_dict = None
    
    elif args.DM == "VFLMonitor":
        # ----------------  將 bank 平均、序列化  ----------------
        from pathlib import Path
        out_dir = Path("VFLMonitor")
        out_dir.mkdir(exist_ok=True)
        bank_path = os.path.join(out_dir, 'vflmonitor_bank.pt')
        fb_builder.save(bank_path)     # 內部自動 finalize→平均→torch.save
        print(f"[VFLMonitor] Feature bank saved to {bank_path}")
        bank = torch.load(bank_path)
        print(bank[0][0].shape)   # 應顯示 e.g. torch.Size([1205, 256])
        all_shapley_outer_dict = None 
        # -------------------------------------------------------

    else:
        all_shapley_outer_dict = None        # 確保至少給一個值
        copur = None
    
    #time 紀錄訓練時長
    t1 = _now()
    timing_train["model_s"] += (t1 - t0)
    
    defense_output = {
        "defense_model": defense_model if args.DM in ["VFLIP","SHAP_MAE","SHAP_MAE_D","NEURAL_CLEANSE","CoPur"] else None,
        "defense_models": None,
        "static_class_dict": static_class_dict if args.DM in ["SHAP_MAE","SHAP_MAE_D"] else None,
        "vec_dict_last_epoch": vec_dict_last_epoch if args.DM in ["SHAP_MAE","SHAP_MAE_D"] else None,
        "y_in_last_epoch": y_in_last_epoch if args.DM in ["SHAP_MAE","SHAP_MAE_D"] else None
    }
    
    n = timing_train["n"]
    _save_json(f"timings/train_overall_{args.DM}_P{args.P}.json", {
        "samples": n,
        "total_model_s": timing_train["model_s"],
        "total_mae_s": timing_train["mae_s"],
        "total_shapley_s": timing_train["shapley_s"],
        "avg_model_ms_per_sample": (timing_train["model_s"] / n) * 1000,
        "avg_mae_ms_per_sample": (timing_train["mae_s"] / n) * 1000,
        "avg_shapley_ms_per_sample": (timing_train["shapley_s"] / n) * 1000
    })
    
    return model, attacker_list, attacker_target_vec_dict, active_party, passive_party, defense_output, all_shapley_outer_dict

# ── 離線 MAE 訓練函式 ──
def train_MAE_offline(defense_model, Htrain_tensor, args, epochs_MAE=None, batch_size_MAE=500):
    """
    離線訓練 MAE 模組
    Htrain_tensor: [N_total, P * bottomModel_output_dim]
    論文中對於 CIFAR10、CINIC10、NUS-WIDE、BM 設 epochs=20，Imagenette 設 epochs=50
    """
    device = args.device
    defense_model.to(device)
    defense_model.train_mode = True
    batch_size_MAE = args.batch_size
    
    mu = Htrain_tensor.mean(dim=0, keepdim=True)   # shape = [1, D]
    std = Htrain_tensor.std(dim=0, keepdim=True) + 1e-5  # 避免除到 0
    defense_model.set_Normalization(mu,std)
    
    if epochs_MAE is None:
        if args.D == "Imagenette":
            epochs_MAE = 50
        else:
            epochs_MAE = 20
    # 使用 Adam 優化器，此處學習率以論文 N-1 to 1 策略 lr=0.01 為例
    optimizer = optim.Adam(defense_model.parameters(), lr=0.1, weight_decay=1e-4, eps=1e-4) #TIFS 8 人嘗試
    # optimizer = optim.Adam(defense_model.parameters(), lr=0.05, weight_decay=1e-4, eps=1e-4)
    dataset_MAE = torch.utils.data.TensorDataset(Htrain_tensor)
    dataloader_MAE = torch.utils.data.DataLoader(dataset_MAE, batch_size=batch_size_MAE, shuffle=True)

    # 初始化 VerticalFLMAELoss（來自 defense_model_structure.py）
    # vertical_loss_fn = dms.VerticalFLMAELoss()
    for e in range(epochs_MAE):
        epoch_loss = 0.0
        for batch in dataloader_MAE:
            batch_x = batch[0].to(device).detach()  # detach 以切斷之前的圖
            mae_input = batch_x.clone()  # clone 出一份獨立的張量
            MAE_output = defense_model(mae_input)
            MAE_loss = defense_model.loss(MAE_output)
            # loss = vertical_loss_fn(model_output)
            optimizer.zero_grad()
            MAE_loss.backward()  # 這裡不需要 retain_graph=True
            optimizer.step()
            epoch_loss += MAE_loss.item()
        print("[MAE Offline] Epoch {}/{}: loss = {:.4f}".format(e+1, epochs_MAE, epoch_loss/len(dataloader_MAE)))
    defense_model.train_mode = False
    # defense_model.set_adaptive_threshold(Htrain_tensor)
    return defense_model

@torch.no_grad()
def collect_thresholds_into_mae(
        defense_model: dms.MAE,
        Htrain_tensor: torch.Tensor,
        args,
        save_path: str | None = None,   # ← None 表示不落檔
        subsample_ratio: float = 0.7,
        rho: float = 2.0,
        batch_size: int = 256):

    device, P, D_perP = args.device, args.P, args.bottomModel_output_dim
    defense_model.eval()

    # ======= 1. optional subsample =======
    K_all = Htrain_tensor.size(0)
    if subsample_ratio < 1.0:
        idx = torch.randperm(K_all, device=Htrain_tensor.device)[: int(K_all*subsample_ratio)]
        H_use = Htrain_tensor[idx]
    else:
        H_use = Htrain_tensor

    loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(H_use),
                batch_size=batch_size, shuffle=False)

    scores = [ [] for _ in range(P) ]
    eyeP   = torch.eye(P, device=device)
    mu, std = defense_model.mu, defense_model.std
    norm_en = defense_model.normalize

    for (xb,) in loader:
        xb = xb.to(device)
        x_in = (xb - mu)/std if norm_en else xb
        x_in = x_in.view(x_in.size(0), -1)

        for j in range(P):
            mj = eyeP[j:j+1].repeat(x_in.size(0),1).repeat_interleave(D_perP, dim=1)
            recon = defense_model.decoder(defense_model.encoder(x_in * mj))
            for i in range(P):
                if i == j: continue
                st, ed = i*D_perP, (i+1)*D_perP
                diff = recon[:,st:ed] - x_in[:,st:ed]
                scores[i].append(diff.norm(2, dim=1).cpu())

    adaptive = {}
    if P ==2:
        rho = 2.0
    elif P==4:
        rho = 1.2
    else: 
        rho = 1.0
    for i in range(P):
        s = torch.cat(scores[i])
        mu_i, sigma_i = s.mean().item(), s.std(unbiased=False).item()
        adaptive[i]   = mu_i + rho * sigma_i

    # ======= 2. 寫入 MAE 物件 =======
    defense_model.adaptive_thresholds = adaptive
    print("[VFLIP] thresholds stored inside MAE.adaptive_thresholds")

    # ======= 3. 可選：另存檔 =======
    if save_path is not None:
        torch.save(adaptive, save_path)
        print(f"           … and serialized to {save_path}")

    return adaptive

def train_SHAP_MAE_offline(defense_model, Htrain_tensor, args, epochs_MAE=None, batch_size_MAE=500):
    """
    離線訓練 MAE 模組
    Htrain_tensor: [N_total, P * bottomModel_output_dim]
    論文中對於 CIFAR10、CINIC10、NUS-WIDE、BM 設 epochs=20，Imagenette 設 epochs=50
    """
    device = args.device
    defense_model.to(device)
    defense_model.train_mode = True
    batch_size_MAE = args.batch_size
    
    mu = Htrain_tensor.mean(dim=0, keepdim=True)   # shape = [1, D]
    std = Htrain_tensor.std(dim=0, keepdim=True) + 1e-5  # 避免除到 0
    defense_model.set_Normalization(mu,std)
    
    if epochs_MAE is None:
        if args.D == "Imagenette":
            epochs_MAE = 50
        else:
            epochs_MAE = 20
    # 使用 Adam 優化器，此處學習率以論文 N-1 to 1 策略 lr=0.01 為例
    optimizer = optim.Adam(defense_model.parameters(), lr=0.05, weight_decay=1e-4, eps=1e-4)
    dataset_MAE = torch.utils.data.TensorDataset(Htrain_tensor)
    dataloader_MAE = torch.utils.data.DataLoader(dataset_MAE, batch_size=batch_size_MAE, shuffle=True)

    # 初始化 VerticalFLMAELoss（來自 defense_model_structure.py）
    # vertical_loss_fn = dms.VerticalFLMAELoss()
    for e in range(epochs_MAE):
        epoch_loss = 0.0
        for batch in dataloader_MAE:
            batch_x = batch[0].to(device).detach()  # detach 以切斷之前的圖
            mae_input = batch_x.clone()  # clone 出一份獨立的張量
            MAE_output = defense_model(mae_input)
            MAE_loss = defense_model.loss(MAE_output)
            # loss = vertical_loss_fn(model_output)
            optimizer.zero_grad()
            MAE_loss.backward()  # 這裡不需要 retain_graph=True
            optimizer.step()
            epoch_loss += MAE_loss.item()
        print("[SHAP_MAE Offline] Epoch {}/{}: loss = {:.4f}".format(e+1, epochs_MAE, epoch_loss/len(dataloader_MAE)))
    defense_model.train_mode = False
    # defense_model.set_adaptive_threshold(Htrain_tensor)
    return defense_model

def train_SHAP_MAE_D_offline(defense_model,
                             concat_embeddings, labels, args):
    assert concat_embeddings.size(0) == labels.size(0), \
        f"N_emb={concat_embeddings.size(0)}  N_lab={labels.size(0)} 長度不符"
    labels = labels.to(torch.long)
    """先跑舊 SHAP_MAE 離線訓練，再補 cosine 統計"""
    defense_model = train_SHAP_MAE_offline(defense_model, concat_embeddings, args)
    # compute_cos_robust_stats(defense_model, concat_embeddings, labels)
    compute_cos_robust_stats(defense_model,
                             concat_embeddings,
                             labels.to(concat_embeddings.device))
    return defense_model

def compute_center_loss(embed, labels, center_layer, parts, d):
    """
    embed : (B, P*d)   -- model._cached_embed
    labels: (B,)       -- y_in (GPU tensor)
    center_layer :     -- model.center_loss (nn.Module)
    parts  : int       -- args.P
    d      : int       -- args.bottomModel_output_dim
    return : scalar tensor
    """
    # 使用 tensor 0 以維持相容裝置與 dtype
    loss_sum = torch.zeros([], device=embed.device)
    for p in range(parts):
        # 不要 detach；梯度要回到底層 bottom-models
        feats_p = embed[:, p*d:(p+1)*d]          # (B, d)
        loss_sum = loss_sum + center_layer(feats_p, labels)
    return loss_sum / parts

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # 輸入參數設定（請依實際需求調整）
    parser.add_argument('--D', type=str, required=True, help='Dataset name, e.g., CIFAR10',choices=["CIFAR10","CINIC10","BM","NUSWIDE"])
    parser.add_argument('--P', type=int, default=2, help='Number of participants')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs for VFL')
    parser.add_argument('--AM', type=str, default="VILLIAN", help='Attack method')
    parser.add_argument('--AN', type=int, default=1, help='Number of attackers')
    parser.add_argument('--AL', type=str, default="C", help='Attacker location setting')
    parser.add_argument('--DM', type=str, default="SHAP_MAE", help='Defense method', choices=['NONE','VFLIP','SHAP_MAE','ABL','NEURAL_CLEANSE','CoPur',"VFLMonitor"])
    parser.add_argument('--act', type=bool, default=True, help='是否有 active party')
    # parser.add_argument('--batch_size', type=int, default=500, help='Batch size')
    parser.add_argument('--bottomModel_output_dim', type=int, default=50, help='Bottom model output dimension')
    # parser.add_argument('--num_class', type=int, default=10, help='Number of classes')
    parser.add_argument('--label', type=int, required=False, default=0, help='the attack target class')
    # 其他參數依需求加入
    
    # --- SHAP_MAE 超參數 -------------------------------------------------
    parser.add_argument('--shap_alpha', type=float, default=None,
                    help='alpha used in Shapley-threshold calculation; '
                         'if None, fall back to the dataset-specific rule')

    # --- ABL hyper-parameters ----------------------------------
    parser.add_argument('--abl_gamma', type=float, default=0.5)
    parser.add_argument('--abl_percent', type=float, default=0.01)
    parser.add_argument('--abl_turn_epoch', type=int, default=20)
    
    # --- CoPur hyper-parameters ---------------------------------
    parser.add_argument('--copur_delta', type=float, default=0.0,
                        help='δ in Eq.(6): AE reconstruction budget; 0 → auto (95-th percentile)')
    parser.add_argument('--copur_tau',   type=float, default=1.0,
                        help='τ in Eq.(6): trade-off between ||h−l||₂,₁ and manifold error')
    parser.add_argument('--copur_iter',  type=int,   default=200,
                        help='Number of gradient steps when solving Eq.(6) during inference')
    
    # --- SHAP_MAE_D hyper-parameters ---------------------------------
    parser.add_argument('--mae_dim', type=int, default=256, help='SHAPMAE_D hidden dim')
    parser.add_argument('--mae_epochs', type=int, default=20,help='SHAPMAE_D offline training epochs')
    parser.add_argument('--k_cos', type=float, required=False, default=0.01, help='the weight of cosine_mad')
    
    # --- VFLMonitor settings ------
    # parser.add_argument('--center_loss_weight', type=float, default=0.0) #0.003
    parser.add_argument('--swd_proj', type=int, default=128)
    parser.add_argument('--tie_rand_seed', type=int, default=42)
    
    # poison_ratio setting
    parser.add_argument('--poison_ratio', type=float, default=None,
                    help='若設定 (0~1)，依比例抽取 steal_id；未設定則沿用原本邏輯')
    
    #BM 特徵分配
    parser.add_argument('--bm_alloc', type=str, default='active_gets_dur',
                    choices=['equal','dur_to_att','active_gets_dur'],
                    help='BM 特徵分配策略')
    
    
    args = parser.parse_args()
    args.batch_size = return_batch_size(args)
    args.start_poison_epoch = Attack.get_start_poison_epoch(args)
    check_poison_ratio(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    if args.DM =="VFLMonitor":
        args.center_loss_weight = 0.003
    # args.batch_size = 500
    print(f"環境中是否有 active party: {args.act}")
    print(f"使用 {args.D} 資料集") 
    data.set_party_num(args.P)
    data.set_bm_alloc(args.bm_alloc)
    # 取得資料（data.get_data 範例）
    (trainloader, testloader, input_size, num_class, trn_x, trn_y, 
     dl_train_set, poisoned_trainloader, val_x, val_y, dl_val_set, poisoned_testloader) = data.get_data(args.D, args.batch_size)
    args.num_class = num_class

    # 執行 poisoning_train（包含主模型訓練及離線 MAE 訓練）
    model, attacker_list, attacker_target_vec_dict, active_party, passive_party, defense_output, train_shap_dict = poisoning_train(
        dl_train_set, poisoned_trainloader, trn_x, trn_y, args
    )
    import eval
    # 測試部分
    args.attacker_list = attacker_list
    args.active_party = active_party
    args.passive_party = passive_party
    clean_test_shap_dict= eval.eval_clean_train(model, testloader, args, defense_output)
    if (args.AM != "None"): 
        poisoned_test_shap_dict = eval.eval_poisoned_train(model, dl_val_set, poisoned_testloader, attacker_target_vec_dict, attacker_list, args, defense_output)
