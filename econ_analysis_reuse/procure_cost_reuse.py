
import os, math
import json

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import time
from copy import deepcopy


class reDimLumber():

    def __init__(self, config={}):

        # Retrieve info
        self.MIN_LEN = config.get("MIN_LEN", 8)
        self.MAX_LEN = config.get("MAX_LEN", 16)
        self.cost_info = config.get("cost_info", 
                                    {
                                        'bm-4':0.54,'bm-6':1.05,'bm-8':1.72,'bm-10':1.89,'bm-12':2.67,'bm-14':8.00,
                                        'bl-4':0.55,'bl-6':0.55,'bl-8':0.62,'bl-10':0.76,'bl-12':0.78,'bl-14':0.89,
                                        'rl-4':0.20,'rl-6':0.18,'rl-8':0.18,'rl-10':0.18,'rl-12':0.17,'rl-14':0.23
                                    }
        )
        ## Connect or not?
        self.p_connect = config.get("p_connect", 0.8)

        # Read data
        self.boq_nc_df = self.read_data(config.get("file_name"), config.get("pjt_name"))

        if config.get("data_size") != 'all':
            self.d_count = config.get("data_size", 100)
        else:
            self.d_count = len(self.boq_nc_df)

    def read_data(self, f_name, p_name):
        # 1. Read a boq.CSV file
        boq_df = pd.read_csv(f'./{f_name}').iloc[:, :6]   # Leaving only the necessary clmns

        # 2. Extract the project of interest
        boq_nc_df = boq_df[boq_df['Family'] == p_name]
        boq_nc_df = boq_nc_df.sort_values(by=['Level'])
        # print(f'Project: {p_name}, Top 3 rows')
        # print(boq_nc_df.head(3))

        return boq_nc_df
    
    def run(self):

        # Plain procurement
        self.vanila_procurement()
        plain_qto = self.summary_std_df
        plain_cutoff = self.summary_cutoff_df

        self.plain_mtrl_cost, self.plain_labor_cost = self.est_cost()
        self.plain_total_cost = round(self.plain_mtrl_cost + self.plain_labor_cost, 2)
        d = {"plain": [self.plain_mtrl_cost, self.plain_labor_cost, 0, self.plain_total_cost]}
        print('Plain procurement & cost analysis --done')
        
        # Cutoff-in-the-loop
        self.reuse_procurement()
        reuse_qto = self.summary_std_df
        reuse_cutoff = self.summary_cutoff_df
        
        self.est_cost_reuse()
        d["reuse"] = [self.cost_mtrl_sum, self.cost_labor_sum, self.cost_handling_sum, self.cost_total_sum]
        print('Cutoff in-the-loop procurement & cost analysis --done\n')

        # Final export
        df_cost = pd.DataFrame(data=d, index=["Material Total", "Labor Total", "Additional Labor Total", "Sum"])
        timestamp = time.strftime("%m%d_%H-%M")
        with pd.ExcelWriter(f"./result_{timestamp}.xlsx") as writer:
            df_cost.to_excel(writer, sheet_name="cost")
            plain_qto.to_excel(writer, sheet_name="plain_qto")
            plain_cutoff.to_excel(writer, sheet_name="plain_cutoff")
            reuse_qto.to_excel(writer, sheet_name="re_total_qto")
            reuse_cutoff.to_excel(writer, sheet_name="re_cutoff")
            self.summary_reuse_df.to_excel(writer, sheet_name="reused_qto")
        print(df_cost)

    def reset(self):
        # Extract
        df_sample = self.sample_data(self.d_count)
        self.sampled_boq_df = deepcopy(df_sample)

        self.summary_std_df = pd.DataFrame(columns=['Lv','T','W','L','Count'])
        self.summary_cutoff_df = pd.DataFrame(columns=['Lv','T','W','L','Count'])
        self.summary_reuse_df = pd.DataFrame(columns=['Lv','Lv_Reused','T','W','L','Count'])

        self.cost_mtrl_sum = 0
        self.cost_labor_sum = 0
        self.cost_handling_sum = 0
        self.cost_saving_sum = 0
        self.cost_total_sum = 0
    
    def vanila_procurement(self):

        self.reset()

        for _, row in self.sampled_boq_df.iterrows():
            # (1) Prep
            len_std, len_remainder = {}, {}

            # (2) Plan by length
            len_org = int(row['Length (ft)'])

            ## Short lumber
            if len_org < self.MIN_LEN:
                len_std[self.MIN_LEN] = 1
                len_remainder[self.MIN_LEN-len_org] = 1

            ## Within a typical range
            elif len_org <= self.MAX_LEN:
                if len_org%2 == 0:
                    len_std[len_org] = 1
                else:
                    len_std[len_org+1] = 1
                    len_remainder[1] = 1
                
            ## Long lumber
            else:
                # Count 16' for the current row
                len_std[self.MAX_LEN] = (len_org//self.MAX_LEN)

                # Process the remainder
                temp = len_org%self.MAX_LEN
                if temp < self.MIN_LEN:
                    len_std[self.MIN_LEN] = 1
                    len_remainder[self.MIN_LEN-temp] = 1
                elif temp%2 == 0:
                    len_std[temp] = 1
                else:
                    len_std[temp+1] = 1
                    len_remainder[1] = 1

            # (3) Update summary df
            target_values = {'Lv': row['Level'], 'T': row['Thickness (in)'], 'W': row['Width (in)']}

            ## Pre-process for 6 Double
            if row['Width (in)'] == '6 Double':
                len_std, len_remainder = self.case_double(len_std, len_remainder)
                target_values['W'] = 6

            ## Standard dimensions
            for ll, lcount in len_std.items():
                self.summary_std_df = self.update_existing(target_values, self.summary_std_df, ll, lcount)

            ## Cutoffs
            for ll, lcount in len_remainder.items():
                self.summary_cutoff_df = self.update_existing(target_values, self.summary_cutoff_df, ll, lcount)
      
    def est_cost(self):
        cost_mtrl_sum, cost_labor_sum = 0, 0
        qto_std_dict = self.qto_by_TxW(self.summary_std_df)
        for w, qty in qto_std_dict.items():
            cost_mtrl_sum += qty*self.cost_info[f'bm-{str(w)}']
            cost_labor_sum += qty*self.cost_info[f'bl-{str(w)}']

        return round(cost_mtrl_sum, 2), round(cost_labor_sum, 2)
        
    def reuse_procurement(self):

        self.reset()

        for _, row in self.sampled_boq_df.iterrows():
            # (1) Prep
            ## len_std & len_remainder = {int(Length): int(Count)}
            ## len_reused = {int(Length): [int(Count), str(Level_reused)]}
            len_std, len_remainder, len_reused = {}, {}, {}

            # (2) Plan by length
            len_org = int(row['Length (ft)'])

            ## Short lumber
            if len_org < self.MIN_LEN:

                # Criteria for reuse
                mask_reuse = self.mask4reuse(row, len_org)

                # Examine whether the matching offcuts exist
                ## Reuse!
                if mask_reuse.any():
                    # Remove Cutoff
                    self.summary_cutoff_df.loc[(np.where(np.array(mask_reuse)==True))[0].tolist()[0],'Count'] -= 1
                    # Add to Reused
                    len_reused[len_org] = [1,self.summary_cutoff_df.loc[(np.where(np.array(mask_reuse)==True))[0].tolist()[0],'Lv']]

                    # If '6 Double', remaining integral part should be addressed
                    if row['Width (in)'] == '6 Double':
                        len_std[self.MIN_LEN], len_remainder[self.MIN_LEN-len_org] = 1, 1

                ## New materials only
                else:
                    if row['Width (in)'] == '6 Double':
                        len_std[self.MIN_LEN], len_remainder[self.MIN_LEN-len_org] = 2, 2
                    else:
                        len_std[self.MIN_LEN], len_remainder[self.MIN_LEN-len_org] = 1, 1

            ## Within a typical range
            elif len_org <= self.MAX_LEN:

                # Examine whether the matching offcuts exist
                mask_reuse = self.mask4reuse(row, len_org)

                ## Reuse!
                if mask_reuse.any():
                    # Remove Cutoff
                    self.summary_cutoff_df.loc[(np.where(np.array(mask_reuse)==True))[0].tolist()[0],'Count'] -= 1
                    # Add to Reused
                    len_reused[temp] = [1,self.summary_cutoff_df.loc[(np.where(np.array(mask_reuse)==True))[0].tolist()[0],'Lv']]
                    # If '6 Double', remaining integral part should be addressed
                    if row['Width (in)'] == '6 Double':
                        len_std[self.MIN_LEN], len_remainder[self.MIN_LEN-temp] = 0.5, 0.5

                ## New materials only
                elif len_org%2 == 0:
                    len_std[len_org] = 1
                else:
                    len_std[len_org+1] = 1
                    len_remainder[1] = 1

                if row['Width (in)'] == '6 Double':
                    len_std, len_remainder = self.case_double(len_std, len_remainder)
                
            ## Long lumber
            else:
                # Count 16' for the current row
                len_std[self.MAX_LEN] = (len_org//self.MAX_LEN)

                # Process the remainder
                temp = len_org%self.MAX_LEN

                # Examine whether the matching offcuts exist
                mask_reuse = self.mask4reuse(row, temp)

                ## Reuse!
                if mask_reuse.any():
                    # Remove Cutoff
                    self.summary_cutoff_df.loc[(np.where(np.array(mask_reuse)==True))[0].tolist()[0],'Count'] -= 1
                    # Add to Reused
                    len_reused[temp] = [1,self.summary_cutoff_df.loc[(np.where(np.array(mask_reuse)==True))[0].tolist()[0],'Lv']]

                    # If '6 Double', remaining integral part should be addressed
                    if row['Width (in)'] == '6 Double':
                        len_std[self.MIN_LEN], len_remainder[self.MIN_LEN-temp] = 0.5, 0.5  # will be applied at the end of else

                ## New Materials Only
                elif temp < self.MIN_LEN:
                    len_std[self.MIN_LEN], len_remainder[self.MIN_LEN-temp] = 1, 1
                elif temp%2 == 0:
                    len_std[temp] = 1
                else:
                    len_std[temp+1] = 1
                    len_remainder[1] = 1

                if row['Width (in)'] == '6 Double':
                    len_std, len_remainder = self.case_double(len_std, len_remainder)

            # (3) Update summary df
            target_values = {'Lv': row['Level'], 'T': row['Thickness (in)'], 'W': row['Width (in)']}

            ## Pre-process for 6 Double
            if row['Width (in)'] == '6 Double':
                target_values['W'] = '6'

            ## Standard dimensions
            for ll, lcount in len_std.items():
                self.update_existing(target_values, self.summary_std_df, ll, lcount)

            ## Cutoffs
            for ll, lcount in len_remainder.items():
                self.update_existing(target_values, self.summary_cutoff_df, ll, lcount)

            # Reused
            for ll, lval in len_reused.items():
                lcount, llv = lval
                target_values['Lv'] = llv
                self.update_existing(target_values, self.summary_reuse_df, ll, lcount, row['Level'])

        self.summary_reuse_df = self.summary_reuse_df.loc[self.summary_reuse_df['Count']>0].reset_index(drop=True)
    
    def est_cost_reuse(self):
        
        # df will be self.summary_std_df
        cost_mtrl_sum, cost_labor_sum, cost_handling_sum = 0, 0, 0

        # Standard dim-lumber
        qto_std_dict = self.qto_by_TxW(self.summary_std_df)
        for w, qty in qto_std_dict.items():
            # Calc cost
            cost_mtrl_sum += qty*self.cost_info[f'bm-{str(w)}']
            cost_labor_sum += qty*self.cost_info[f'bl-{str(w)}']
            # Stack costs
            cost_std_sum += (cost_mtrl_sum + cost_labor_sum)

        # Reused dim-lumber
        qto_reuse_dict = self.qto_by_TxW(self.summary_reuse_df)
        for w, qty in qto_reuse_dict.items():
            ## Calc cost
            cost_handling_sum += qty*self.cost_info[f'rl-{str(w)}']
            
        self.cost_mtrl_sum += round(cost_mtrl_sum, 2)
        self.cost_labor_sum += round(cost_labor_sum, 2)
        self.cost_handling_sum += round(cost_handling_sum, 2)
        self.cost_total_sum += round(cost_mtrl_sum + cost_labor_sum + cost_handling_sum, 2)
    
    def case_double(self, len_std, len_remainder):
        
        doubled_len_std = {key: value*2 for key, value in len_std.items()}
        doubled_len_remainder = {key: value*2 for key, value in len_remainder.items()}

        return doubled_len_std, doubled_len_remainder
    
    def update_existing(self, target_values, existing_df, ll, lcount, level='none'):

        # Prep
        if level == 'none':
            target_values['L'] = ll
            mask = (existing_df[list(target_values)] == pd.Series(target_values)).all(axis=1)

        else:
            target_values['L'], target_values['Lv_Reused'] = ll, level
            mask = (existing_df[list(target_values)] == pd.Series(target_values)).all(axis=1)

        # Does {LV, T, W, L} combination already exist in the df?
        ## Exist
        if mask.any():
            existing_df.loc[(np.where(np.array(mask)==True))[0].tolist()[0],'Count'] += lcount
        ## Not exist
        else:
            target_values['Count'] = lcount
            existing_df.loc[len(existing_df)] = target_values

        return existing_df
    
    def mask4reuse(self, row, length):
        # target_values = {'T': row['Thickness (in)'], 'W': row['Width (in)'], 'L': row['Length (ft)']}
        if row['Width (in)'] == '6 Double':
            target_values = {'T': row['Thickness (in)'], 'W': '6', 'L': length}
        else:
            target_values = {'T': row['Thickness (in)'], 'W': row['Width (in)'], 'L': length}
        
        mask_reuse = ((self.summary_cutoff_df[list(target_values)] == pd.Series(target_values)).all(axis=1) 
                      & (self.summary_cutoff_df['Lv'] < row['Level']) 
                      & (self.summary_cutoff_df['Count'] > 0))
        
        return mask_reuse

    def qto_by_TxW(self, df):

        # Calculate the total length of each row
        df['Total_L'] = df['L']*df['Count']

        # Leave only T and W info
        summed_series = df.groupby(['W'])['Total_L'].sum()

        return summed_series.to_dict()
    
    def sample_data(self, data_count):

        return self.boq_nc_df.head(data_count)
