import data
import time
import os
import math
import itertools
from functools import reduce
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import cupy as cp
from cal_centers import train_mean_shift

class NeedMoreEpochs(Exception):
    pass
class TotalNumberWrong(Exception):
    pass

def get_start_defense_epoch(args):
    if args.DM == "VFLIP":
        s_d_e = args.epochs-20
    if args.DM == "SHAP_MAE":
        s_d_e = args.epochs-20
    if s_d_e <= 0:
        raise NeedMoreEpochs("Please make sure the number of epochs more than 20")
    return s_d_e

def get_shapley_samples(trn_x, trn_y, num_sample, label):
    targets = []
    t = label
    print(f"正在抽樣 label 為 {t} 的 sample")
    for idx in range(len(trn_y)):
        if trn_y[idx] == t:
            targets.append(trn_x[idx]['id'])

    sample_id = random.sample(targets, num_sample)
    sample_x = []
    label = []
    for idx in sample_id:
        sample_x.append(trn_x[idx])
        label.append(trn_y[idx])
    shapley_set = data.CIFAR10(sample_x,label)
    shapley_id = torch.tensor(sample_id)
    return shapley_set, shapley_id

@torch.no_grad()
def do_shapley(model, vec, y_in, args):
    # 每個類別分檔
    # 創建嵌套字典
    shapley_outer_dict = {outer_key: {} for outer_key in range(args.num_class)}  # 外層字典
    returned_shapley_outer_dict = {outer_key: {} for outer_key in range(args.num_class)}
    class_count_into_for_number_list = [0 for _ in range(args.num_class)]
    count_number_list = [0 for _ in range(args.num_class)]
    vec_base = vec.detach().clone()  # 確保 vec_base 不被修改
    num_combinations = 2**args.P
    
    for i in range(args.num_class):
        # print(f"現在正在計算 label 為 {i} shapley value")
        with torch.inference_mode(mode=True):
            model.eval()
            for p in range(args.P+1):
                combinations = []
                combinations.extend(list(itertools.combinations(list(range(args.P)),p)))
                for combination in combinations:
                    # print(combination)
                    vec_clone = vec_base.clone()
                    #將元素以 args.bottomModel_output_dim 個為一單位
                    indices = (torch.arange(args.bottomModel_output_dim * args.P) // args.bottomModel_output_dim)
                    combination_tensor = torch.tensor(combination, dtype=indices.dtype, device=indices.device)
                    # 使用 isin 判斷 mask
                    mask = torch.isin(indices, combination_tensor).view(1, -1)  # 將 mask 轉為 2D
                    mask = mask.expand(vec.size(0), -1)  # 匹配 vec 的形狀
                    vec_clone[mask] = 0
                    
                    label_mask = (y_in == i)
                    pred = model.top(vec_clone[label_mask])
                    softmax = nn.Softmax(dim=1)
                    pred_softmax = softmax(pred)
                    log_pred_softmax = pred_softmax[:,i]
                    
                    count_number_list[i]+= log_pred_softmax.shape[0]
                    
                    if class_count_into_for_number_list[i]==0: #初始化
                        shapley_inner_key = list(range(args.P))+["Prob"]  # 內層字典的鍵
                        shapley_outer_dict[i] = {key: [1] for key in shapley_inner_key if key != "Prob"}  # 先初始化整個字典
                        
                        # 確保 "Prob" 欄位有 `num_combinations` 行的 Tensor
                        shapley_outer_dict[i]["Prob"] = [torch.empty(0, device=vec.device)] * num_combinations

                        pred_base = model.top(vec_base[label_mask])
                        pred_softmax_base = softmax(pred_base)
                        log_pred_softmax_base = pred_softmax_base[:, i]
                        shapley_outer_dict[i]["Prob"][0] = log_pred_softmax_base  # 設定初始值
                        
                    else:
                        if len(shapley_outer_dict[i][0]) < num_combinations: #隨便取一個來抓現在有多少紀錄
                            for participant in range(args.P):
                                if participant in combination:
                                    shapley_outer_dict[i][participant].append(0)
                                else:
                                    shapley_outer_dict[i][participant].append(1)
                                    
                            row_number= class_count_into_for_number_list[i]%num_combinations
                            
                            shapley_outer_dict[i]["Prob"][row_number] = torch.cat(
                                (shapley_outer_dict[i]["Prob"][row_number], log_pred_softmax), dim=0
                            )
                            
                        else:
                            for m in range(len(shapley_outer_dict[i][0])):
                                count = 0
                                for part in range(args.P):
                                    if (part in combination) != (shapley_outer_dict[i][part][m]):
                                        # print(f"判定結果為 {part in combination} 和 {shapley_outer_dict[i][part][m]}")
                                        count+=1
                                if count == args.P:
                                    row_number= class_count_into_for_number_list[i]%num_combinations
                                    zero_tensor = torch.zeros_like(log_pred_softmax)
                            
                                    shapley_outer_dict[i]["Prob"][row_number] = torch.cat(
                                        (shapley_outer_dict[i]["Prob"][row_number], zero_tensor), dim=0
                                    )
                    class_count_into_for_number_list[i]+=1

        
        returned_shapley_inner_key = list(range(args.P))
        returned_shapley_outer_dict[i] = {key: torch.empty(0, device=vec.device) for key in returned_shapley_inner_key}
        for m in range(len(shapley_outer_dict[i][0])):
            shapley_outer_dict[i]["Prob"][m] = shapley_outer_dict[i]["Prob"][m].cpu()
        shapley_dataframe = pd.DataFrame.from_dict(shapley_outer_dict[i])

        list_each_participant_shapley_in_this_batch = calculate_each_participant_shap_in_one_batch(shapley_dataframe,args)
        for p in range(args.P):

            returned_shapley_outer_dict[i][p] = torch.cat(
                (returned_shapley_outer_dict[i][p], list_each_participant_shapley_in_this_batch[p].to(args.device)), dim=0)
        
    model.train()
    return returned_shapley_outer_dict

def do_every_class_for_every_sample_shapley(model, vec, args):
    '''
    測試時專用
    input : vec
    output : 這一 batch 每個 sample 在每個 class 的貢獻度
    '''
    # 每個類別分檔
    # 創建嵌套字典
    shapley_outer_dict = {outer_key: {} for outer_key in range(args.num_class)}  # 外層字典
    returned_shapley_outer_dict = {outer_key: {} for outer_key in range(args.num_class)}
    class_count_into_for_number_list = [0 for _ in range(args.num_class)]
    count_number_list = [0 for _ in range(args.num_class)]
    vec_base = vec.detach().clone()  # 確保 vec_base 不被修改
    num_combinations = 2**args.P
    
    for i in range(args.num_class):
        # print(f"現在正在計算 label 為 {i} shapley value")
        with torch.inference_mode(mode=True):
            model.eval()
            for p in range(args.P+1):
                combinations = []
                combinations.extend(list(itertools.combinations(list(range(args.P)),p)))
                for combination in combinations:
                    # print(combination)
                    vec_clone = vec_base.clone()
                    #將元素以 args.bottomModel_output_dim 個為一單位
                    indices = (torch.arange(args.bottomModel_output_dim * args.P) // args.bottomModel_output_dim)
                    combination_tensor = torch.tensor(combination, dtype=indices.dtype, device=indices.device)
                    # 使用 isin 判斷 mask
                    mask = torch.isin(indices, combination_tensor).view(1, -1)  # 將 mask 轉為 2D
                    mask = mask.expand(vec.size(0), -1)  # 匹配 vec 的形狀
                    vec_clone[mask] = 0
                    
                    # label_mask = (y_in == i)
                    pred = model.top(vec_clone)
                    softmax = nn.Softmax(dim=1)
                    pred_softmax = softmax(pred)
                    log_pred_softmax = pred_softmax[:,i]
                    
                    count_number_list[i]+= log_pred_softmax.shape[0]
                    
                    if class_count_into_for_number_list[i]==0: #初始化
                        shapley_inner_key = list(range(args.P))+["Prob"]  # 內層字典的鍵
                        shapley_outer_dict[i] = {key: [1] for key in shapley_inner_key if key != "Prob"}  # 先初始化整個字典
                        
                        # 確保 "Prob" 欄位有 `num_combinations` 行的 Tensor
                        shapley_outer_dict[i]["Prob"] = [torch.empty(0, device=vec.device)] * num_combinations

                        pred_base = model.top(vec_base)
                        pred_softmax_base = softmax(pred_base)
                        log_pred_softmax_base = pred_softmax_base[:, i]
                        shapley_outer_dict[i]["Prob"][0] = log_pred_softmax_base  # 設定初始值
                        
                    else:
                        if len(shapley_outer_dict[i][0]) < num_combinations: #隨便取一個來抓現在有多少紀錄
                            for participant in range(args.P):
                                if participant in combination:
                                    shapley_outer_dict[i][participant].append(0)
                                else:
                                    shapley_outer_dict[i][participant].append(1)
                                    
                            row_number= class_count_into_for_number_list[i]%num_combinations
                            shapley_outer_dict[i]["Prob"][row_number] = torch.cat(
                                (shapley_outer_dict[i]["Prob"][row_number], log_pred_softmax), dim=0
                            )
                        else:

                            for m in range(len(shapley_outer_dict[i][0])):
                                count = 0
                                for part in range(args.P):
                                    if (part in combination) != (shapley_outer_dict[i][part][m]):
                                        count+=1
                                if count == args.P:
                                    row_number= class_count_into_for_number_list[i]%num_combinations
                                    shapley_outer_dict[i]["Prob"][row_number] = torch.cat(
                                        (shapley_outer_dict[i]["Prob"][row_number], log_pred_softmax), dim=0
                        )
                    class_count_into_for_number_list[i]+=1

        
        returned_shapley_inner_key = list(range(args.P))
        returned_shapley_outer_dict[i] = {key: torch.empty(0, device=vec.device) for key in returned_shapley_inner_key}
        for m in range(len(shapley_outer_dict[i][0])):
            shapley_outer_dict[i]["Prob"][m] = shapley_outer_dict[i]["Prob"][m].cpu()
        shapley_dataframe = pd.DataFrame.from_dict(shapley_outer_dict[i])
        list_each_participant_shapley_in_this_batch = calculate_each_participant_shap_in_one_batch(shapley_dataframe,args)
        for p in range(args.P):

            returned_shapley_outer_dict[i][p] = torch.cat(
                (returned_shapley_outer_dict[i][p], list_each_participant_shapley_in_this_batch[p].to(args.device)), dim=0)
            # print(f"回傳後蒐集完的 dict 為:{returned_shapley_outer_dict[i][p]}")
    return returned_shapley_outer_dict

def factorial(n):
    if n == 0:
        return 1
    else:
        return n*factorial(n-1)

#原版，沒有套用 P-shapley 

import pandas as pd
import numpy as np
import torch
from math import factorial
from typing import List, Union

TensorLike = Union[torch.Tensor, np.ndarray, float]

def calculate_each_participant_shap_in_one_batch(
        shapley_dataframe: pd.DataFrame,
        args,
        normalize: str | None = None,
        eps: float = 1e-12
    ):
    """
    計算單一 batch 中每位參與者的 Shapley Value。
    
    Parameters
    ----------
    shapley_dataframe : pd.DataFrame
        內容同前；'Prob' 欄可為 tensor、ndarray 或 float。
    args : Namespace
        至少需包含 args.P (參與者總數)。
    normalize : {"proportion", None}
        - None         : 直接回傳 raw Shapley。
        - "proportion" : 先將所有 Shapley 值平移至非負，再以總和做比例化。
    eps : float
        防止除 0 的極小常數。
    """
    # ---------- (0) 推斷向量維度並初始化 ----------
    first_vec = shapley_dataframe["Prob"].iloc[0]
    if isinstance(first_vec, torch.Tensor):
        zero_vec = torch.zeros_like(first_vec, dtype=torch.float64)
    elif isinstance(first_vec, np.ndarray):
        zero_vec = np.zeros_like(first_vec, dtype=np.float64)
    else:                         # scalar
        zero_vec = 0.0

    shap_values: List[TensorLike] = [
        zero_vec.clone() if isinstance(zero_vec, torch.Tensor)
        else (zero_vec.copy() if isinstance(zero_vec, np.ndarray) else 0.0)
        for _ in range(args.P)
    ]

    # ---------- (1) 累積每位參與者的 Shapley ----------
    for participant in shapley_dataframe.columns:
        if participant == "Prob":
            continue

        df_in  = shapley_dataframe[shapley_dataframe[participant] == 1].reset_index(drop=True)
        df_out = shapley_dataframe[shapley_dataframe[participant] == 0].reset_index(drop=True)

        df_in  = df_in .drop(participant, axis=1).copy()
        df_out = df_out.drop(participant, axis=1).copy()
        df_in ["WithACC"]    = df_in .pop("Prob")
        df_out["WithoutACC"] = df_out.pop("Prob")

        other_cols = list(df_in.columns); other_cols.remove("WithACC")
        sheet = pd.merge(df_in, df_out, on=other_cols)

        n = args.P
        for _, row in sheet.iterrows():
            s = (row[other_cols] == 1).sum() + 1
            weight = factorial(s-1) * factorial(n-s) / factorial(n)
            contrib = row["WithACC"] - row["WithoutACC"]
            shap_values[int(participant)] += weight * contrib

    # ---------- (2) proportion 正規化（可選） ----------
    if normalize == "proportion":
        # 堆疊成 (P,d) 或 (P,) 方便向量化
        stacked = torch.stack(shap_values) if isinstance(shap_values[0], torch.Tensor) \
                  else np.stack(shap_values, axis=0)

        # (2‑A) 平移：確保所有值 ≥ 0
        min_val = stacked.min(dim=0).values if isinstance(stacked, torch.Tensor) else stacked.min(axis=0)
        shift   = (-min_val).clamp(min=0) + eps if isinstance(stacked, torch.Tensor) \
                  else np.clip(-min_val, a_min=0, a_max=None) + eps
        shifted = stacked + shift                    # broadcasting

        # (2‑B) 計算佔比
        denom = shifted.sum(dim=0) if isinstance(shifted, torch.Tensor) else shifted.sum(axis=0)
        proportion = shifted / denom                 # broadcasting

        shap_values = [v for v in proportion]        # 轉回 list

    return shap_values

def calculateAndDraw_distribution_of_shapley_value(all_shapley_outer_dict, args, tag=0, draw=False, id_in_class=None, steal_id_in_class=None):
    static_class_dict = {outer_key: {} for outer_key in range(args.num_class)}
    static_participant_key = list(range(args.P))
    # 新增 q1 與 q3 四分位數

    for i in range(args.num_class):
        static_class_dict[i] = {key: {} for key in static_participant_key}
        
        if i == args.label and tag == 0:
            steal_index_in_class = [id_ in steal_id_in_class for id_ in id_in_class]
            steal_mask = torch.tensor(steal_index_in_class, dtype=torch.bool)
            non_steal_mask = ~steal_mask
        
        for j in range(args.P):
            values = all_shapley_outer_dict[i][j].cpu().numpy()
            mean_value = np.mean(values)
            median_value = np.median(values)
            std_dev = np.std(values)
            variance = np.var(values)
            mad_value = np.median(np.abs(values - median_value))
            # 計算第一與第三四分位數
            q1_value = np.percentile(values, 25)
            q3_value = np.percentile(values, 75)
            
            up_distance = q3_value-median_value
            down_distance = median_value-q1_value
            
            ratio = math.log(args.P, 2)
            # ratio = 1
            
            # ❶ 假如使用者 CLI 明確給 alpha，就直接採用
            if args.shap_alpha is not None:
                alpha = args.shap_alpha
            # ❷ 否則沿用你原本依資料集決定的預設
            elif args.D == "CIFAR10":
                if args.AM == "TIFS":
                    alpha = 1.7
                elif args.AM == "VILLIAN":
                    alpha = 2.0
            elif args.D == "CINIC10":
                if args.AM == "TIFS":
                    alpha = 1.3
                elif args.AM == "VILLIAN":
                    alpha = 1.5
            else: # BM
                if args.AM == "TIFS":
                    alpha = 1.3
                elif args.AM == "VILLIAN":
                    alpha = 1.5
            
            # 自創
            up_threshold = median_value + up_distance/(ratio*alpha)
            down_threshold = median_value - down_distance/(ratio*alpha)
            
            
            
            static_class_dict[i][j] = {
                "mean": mean_value,
                "median": median_value,
                "std": std_dev,
                "variance": variance,
                "mad": mad_value,
                "q1": q1_value,
                "q3": q3_value,
                "up_threshold": up_threshold,
                "down_threshold": down_threshold
            }
            
            if draw:
                plt.figure(figsize=(10, 6))
                sns.histplot(values, bins=50, kde=True, color='skyblue')
                
                # 僅標示中位數、median_value ± mad_value 以及第一與第三四分位數
                plt.axvline(median_value, color='blue', linestyle='--', label=f'Median: {median_value:.4f}')
                plt.axvline(down_threshold, color='green', linestyle='-.', label=f'Down Threshold: {down_threshold:.4f}')
                plt.axvline(up_threshold, color='green', linestyle='-.', label=f'Up Threshold: {up_threshold:.4f}')
                
                plt.title(f'Shapley Value Distribution (Class {i}, P {j})')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.legend()
                if tag == 0:
                    filename = f'train_class_{i}_P_{j}_distribution.png'
                elif tag == 1:
                    filename = f'test_class_{i}_P_{j}_distribution.png'
                else:
                    filename = f'poisoned_class_{i}_P_{j}_distribution.png'
                plt.savefig(filename)
                plt.close()
                
                if tag == 0 and i == args.label:
                    steal_values = all_shapley_outer_dict[i][j][steal_mask].cpu().numpy()
                    non_steal_values = all_shapley_outer_dict[i][j][non_steal_mask].cpu().numpy()
                    
                    plt.figure(figsize=(10, 6))
                    sns.histplot(steal_values, bins=50, kde=True, color='red', label='Poisoned')
                    sns.histplot(non_steal_values, bins=50, kde=True, color='blue', label='Clean')
                    plt.title(f'Steal vs Non-Steal Samples - Class {i} P {j}')
                    plt.xlabel('Value')
                    plt.ylabel('Frequency')
                    plt.legend()
                    plt.savefig(f'class_{i}_P_{j}_steal_vs_non_steal.png')
                    plt.close()
    if tag == 0:
        return static_class_dict
    else:
        return 0

def cal_shapley_distribution(counter_dict, args):
    """
    計算當參與者的 shapley value count 達到標準（例如9）時，
    統計同一筆資料中其他參與者的 count 分佈：
      - 針對所有其他參與者（分佈數量）
      - 分別針對正常參與者與其他攻擊者（除了該攻擊者本身）
    並計算各組別在各個 count 值（例如 0 到標準值）的平均、最大、最小值以及比例（相對於群體總人數）。

    輸入參數：
      counter_dict: dict，鍵為參與者編號，值為長度為 N 的 torch.Tensor（N例如10000），數值代表該筆資料的 count
      args: 具有下列屬性：
            - P: 參與者總數
            - attacker_list: 攻擊者編號列表
            - num_class: 類別數 (例如 10，此時標準值為 num_class - 1，即9)
    """
    attacker_list = args.attacker_list
    total_participants = args.P
    standard = args.num_class - 1  # 例如 9
    counter_np = {p: counter_dict[p].cpu().numpy() for p in range(total_participants)}
    num_samples = len(next(iter(counter_np.values())))
    
    for p in range(total_participants):
        mask = (counter_np[p] == standard)
        indices = np.where(mask)[0]  # 被偵測到的 sample index
        num_detected = len(indices)
        perc = round(num_detected / num_samples * 100, 2)
        # print(f"參與者 {p} 有 {num_detected} 筆資料達到標準 (達標比例: {perc}%)")
        
        if num_detected == 0:
            continue
        
        overall_counts = defaultdict(int)
        normal_counts = defaultdict(int)
        other_attacker_counts = defaultdict(int)
        
        for idx in indices:
            for other_p in range(total_participants):
                if other_p == p:
                    continue
                cnt_val = int(counter_np[other_p][idx])
                cnt_val = max(0, min(cnt_val, standard))
                overall_counts[cnt_val] += 1
                if other_p in attacker_list:
                    other_attacker_counts[cnt_val] += 1
                else:
                    normal_counts[cnt_val] += 1
        
        total_normal = total_participants - len(attacker_list)
        total_other_attacker = max(len(attacker_list) - 1, 0)
        # print(f"\n統計結果(當參與者 {p} 達到標準時)：")
        
        x_vals = list(range(standard + 1))
        overall_arr = np.array([overall_counts[v] for v in x_vals])
        normal_arr = np.array([normal_counts[v] for v in x_vals])
        attacker_arr = np.array([other_attacker_counts[v] for v in x_vals])
        
        plt.figure(figsize=(10, 6))
        plt.bar(x_vals, attacker_arr, color='red', label='Attacker')
        plt.bar(x_vals, normal_arr, bottom=attacker_arr, color='blue', label='Normal participant')
        plt.xlabel('The setting Shapley Value ')
        plt.ylabel('Nmmber of participants')
        plt.title(f'The Distribution when participant {p} over the Shapley Value threshold')
        plt.legend()
        plt.savefig(f'participant_{p}.png')
        plt.close()

def do_predicted_class_for_every_sample_shapley(model, vec, args):
    '''
    測試時專用，只計算每筆資料預測類別下各參與者的貢獻度
    input : vec
    output : 這一 batch 中，每個參與者在各筆資料預測類別上的 Shapley value（以 list 表示，每個元素為一個 torch.Tensor）
    '''
    returned_preded_shapley_key = list(range(args.P))
    returned_preded_shapley_dict = {key: torch.empty(0, device=vec.device) for key in returned_preded_shapley_key}
    # 先備份輸入資料
    vec_base = vec.detach().clone()
    
    # 取得 base prediction 與每筆資料的預測類別
    with torch.inference_mode():
        model.eval()
        base_pred = model.top(vec_base)
        softmax = nn.Softmax(dim=1)
        base_softmax = softmax(base_pred)
        # 對每個 sample 取最高機率的類別作為預測類別
        predicted_class = base_softmax.argmax(dim=1)  # shape: (batch_size,)
    
    num_combinations = 2**args.P
    # 建立存放所有組合資訊的字典，鍵為各參與者（0 ~ args.P-1）及 "Prob"
    shapley_dict = {key: [] for key in list(range(args.P)) + ["Prob"]}
    
    # 第一組合：不 mask 任何參與者（皆納入），直接使用 base 的預測機率（注意：取每筆 sample 預測類別的機率）
    base_prob = base_softmax[torch.arange(vec.size(0)), predicted_class].cpu()
    shapley_dict["Prob"].append(base_prob)
    for participant in range(args.P):
        shapley_dict[participant].append(1)  # 1 表示該參與者被納入
    
    # 接下來遍歷所有至少 mask 一位參與者的組合
    for p in range(1, args.P+1):  # p 表示 mask 的參與者數量
        for combination in itertools.combinations(range(args.P), p):
            vec_clone = vec_base.clone()
            # 將每個參與者的輸出當作一個區段，利用 indices 決定哪些區段要 mask
            indices = (torch.arange(args.bottomModel_output_dim * args.P, device=vec.device) //
                       args.bottomModel_output_dim)
            combination_tensor = torch.tensor(combination, dtype=indices.dtype, device=indices.device)
            mask = torch.isin(indices, combination_tensor).view(1, -1)
            mask = mask.expand(vec.size(0), -1)
            vec_clone[mask] = 0
            
            with torch.inference_mode():
                pred = model.top(vec_clone)
                pred_softmax = softmax(pred)
            # 取出每筆 sample 在其預測類別上的機率
            sample_indices = torch.arange(vec.size(0), device=vec.device)
            prob_values = pred_softmax[sample_indices, predicted_class].cpu()
            shapley_dict["Prob"].append(prob_values)
            
            # 記錄各參與者是否有納入：若該參與者在 combination 中，代表被 mask，記 0；否則記 1
            for participant in range(args.P):
                flag = 0 if participant in combination else 1
                shapley_dict[participant].append(flag)
    
    # 將整理好的字典轉為 DataFrame，再利用現有的 Shapley value 計算函式
    shapley_dataframe = pd.DataFrame(shapley_dict)
    list_each_participant_shapley = calculate_each_participant_shap_in_one_batch(shapley_dataframe, args)
    for p in range(args.P):
        # print(returned_shapley_outer_dict[i][p])
        # print(list_each_participant_shapley_in_this_batch[p])
        returned_preded_shapley_dict[p] = torch.cat(
            (returned_preded_shapley_dict[p], list_each_participant_shapley[p].to(args.device)), dim=0)
            # print(f"回傳後蒐集完的 dict 為:{returned_shapley_outer_dict[i][p]}")
    return returned_preded_shapley_dict

def how_many_data_over_threshold(preded_shapley_dict, static_class_dict, pred_c, args):
    """
    作用：對一個 batch 資料計算
    回傳：
      - flagged_passive_dict: list，每筆資料有哪些 passive 被標記為 exceed
      - conflict_sample_count: 有 active below + passive exceed 的樣本數
      - attacker_exceed_counts: 每位 attacker exceed threshold 的次數
      - participant_exceed_in_conflict: 每位參與者在 active below 條件下 exceed 的次數
    """
    active_party = args.active_party
    num_samples = len(pred_c)
    passive_party_list = [p for p in range(args.P) if p != active_party]
    attacker_list = args.attacker_list

    flagged_passive_dict = [set() for _ in range(num_samples)]
    conflict_sample_count = 0
    attacker_exceed_counts = {p: 0 for p in attacker_list}
    participant_exceed_in_conflict = {p: 0 for p in range(args.P)}

    for i in range(num_samples):
        label = pred_c[i].item() if isinstance(pred_c[i], torch.Tensor) else int(pred_c[i])
        sample_status = {}

        for p in range(args.P):
            shap_val = preded_shapley_dict[p][i]
            thresholds = static_class_dict[label][p]
            up_threshold = thresholds['up_threshold']
            down_threshold = thresholds['down_threshold']

            if shap_val > up_threshold:
                sample_status[p] = 'exceed'
                if p in attacker_list:
                    attacker_exceed_counts[p] += 1
            elif shap_val < down_threshold:
                sample_status[p] = 'below'
            else:
                sample_status[p] = 'within'

        # 如果 active 是 below，統計 passive 的 exceed 狀況
        if sample_status[active_party] == 'below':
            for p in passive_party_list:
                if sample_status[p] == 'exceed':
                    flagged_passive_dict[i].add(p)
                    participant_exceed_in_conflict[p] += 1
            if len(flagged_passive_dict[i]) > 0:
                conflict_sample_count += 1

    return (
        flagged_passive_dict,
        conflict_sample_count,
        attacker_exceed_counts,
        participant_exceed_in_conflict
    )
    
def detect_how_much_sample_shapley_s0_in_P(all_shapley_outer_dict, args, tag=0):
    '''
    tag:
        0 : train
        1 : clean testset
        2 : poisoned testset
        3 : take the detected index
    '''
    counter_dict = {key: torch.empty(0, device=args.device) for key in list(range(args.P))}
    for j in range(args.P):
        tensor_list=[]
        for i in range(args.num_class):
            if tag != 0:
                # print(f"現在 tag 是:{tag}，在蒐集第{j}個參與者在第{i}類別的資料")
                tensor_list.append(all_shapley_outer_dict[i][j])
        # print(len(tensor_list))
        # print(tensor_list)
        # time.sleep(3)

        if tag != 0: 
            # 檢查每個 Tensor 小於 0 的位置，轉成布林張量
            masks = [tensor < 0 for tensor in tensor_list]
            '''
            一個參與者在每個 sample 有幾次小於 0，回傳 tensor，index 表示sample編號
            '''
            negative_counts = sum(masks)
            counter_dict[j] = torch.cat((counter_dict[j],negative_counts), dim=0)
            # print(len(negative_counts))
            # if tag != 3:
            #     for k in range(args.num_class+1):
            #         print(f"參與者{j}，有{(negative_counts==k).sum().item()}個 sample 在 {k} 個類別貢獻度小於0")
            #     # print(counter_dict)
    if tag ==3:
        return counter_dict
    else:
        cal_shapley_distribution(counter_dict,args)
        return 0