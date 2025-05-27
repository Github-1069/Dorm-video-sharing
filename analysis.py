import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 

class DataAnalysis:
    def __init__(self, db_path):
        self.db_path = db_path
        self.back_df = pd.DataFrame()
        self.df=self.load_data()
        self.lambda_hat = None 

    def load_data(self):
        self.df = pd.read_csv(self.db_path)
        return self.df

    def select_data(self, column_name):
        self.back_df[column_name] = self.df[column_name]
        self.back_df['time'] = self.df['time']
        self.back_name = column_name
        self.back_df = self.back_df.dropna()
        return self.back_df
    
    def count_data(self):
        self.back_df['count'] = range(1, len(self.back_df) + 1) 
        self.back_df.drop(columns=[self.back_name], inplace=True)
        return self.back_df
    
    def count_all(self):
        self.df['count'] = range(1,len(self.df)+1)
        # self.df.drop(columns=['W','P','S'],inplace=True)
        return self.df
    
    def show_data(self):
        plt.figure(figsize=(10, 6))
        plt.step(self.back_df['time'], self.back_df['count'], where='post', label='计数')
        plt.xlabel('时间')
        plt.ylabel('计数')
        plt.title('计数随时间变化的阶梯图')
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def regress(self,select=False):
        """
        拟合泊松过程并检验拟合优度，考虑每天7:00作为起始时间
        """
        if select:
            times = pd.to_datetime(self.back_df['time']).values 
        else:
            times = pd.to_datetime(self.df['time']).values 
        
        time_diffs = []
        for i in range(1, len(times)):
            current_time = pd.Timestamp(times[i])
            prev_time = pd.Timestamp(times[i-1])
            
            if current_time.hour >= 7:
                current_seven_am = current_time.replace(hour=7, minute=0, second=0, microsecond=0)
                if prev_time >= current_seven_am:
                    diff = (current_time - prev_time).total_seconds() / 60
                else:
                    diff = (current_time - current_seven_am).total_seconds() / 60
            else:
                if prev_time.date() == current_time.date() or prev_time.date() == current_time.date() - pd.Timedelta(days=1):
                    diff = (current_time - prev_time).total_seconds() / 60
                else:
                    prev_day_seven_am = (current_time - pd.Timedelta(days=1)).replace(hour=7, minute=0, second=0, microsecond=0)
                    diff = (current_time - prev_day_seven_am).total_seconds() / 60
            
            time_diffs.append(diff)
        
        time_diffs = np.array(time_diffs)
        
        self.lambda_hat = 1 / np.mean(time_diffs)
        
        # 进行卡方检验
        hist, bin_edges = np.histogram(time_diffs, bins='auto', density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 计算理论频数
        bin_width = bin_edges[1] - bin_edges[0]
        theoretical_freq = len(time_diffs) * stats.expon.pdf(bin_centers, scale=1/self.lambda_hat) * bin_width
        mask = (hist > 0) & (theoretical_freq > 0)
        hist = hist[mask]
        theoretical_freq = theoretical_freq[mask]
        
        # 调整理论频数使其总和与观测频数一致
        theoretical_freq = theoretical_freq * (np.sum(hist) / np.sum(theoretical_freq))
        
        chi2_stat, p_value = stats.chisquare(hist, theoretical_freq)
        
        plt.figure(figsize=(12, 6))
        
        plt.hist(time_diffs, bins='auto', density=True, alpha=0.6, label='实际分布')
        
        x = np.linspace(0, max(time_diffs), 100)
        plt.plot(x, stats.expon.pdf(x, scale=1/self.lambda_hat), 
                'r-', label='理论指数分布')
        
        plt.xlabel('时间间隔（分钟）')
        plt.ylabel('概率密度')
        plt.title(f'泊松过程拟合结果\nλ = {self.lambda_hat:.4f} (每分钟), p值 = {p_value:.4f}')
        plt.legend()
        plt.grid(True)
        if not select:
            plt.savefig(f'Aggregate_regress.png')
        else:
            plt.savefig(f'{self.back_name}_regress.png')
        plt.show()
        
        # 自相关检验
        lb_test = acorr_ljungbox(time_diffs, lags=10, return_df=True)
        
        return {
            'lambda_hat': self.lambda_hat,
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'lb_test': lb_test
        }
    
    def estimate(self,select=False,choose='P'):
        if select:
            self.select_data(choose)
            self.count_data()
            results = self.regress(select=True)
            martingale_results = self.martingale_analysis(select=True, choose=choose)
            print("\n鞅分析结果：")
            print(f"t统计量 = {martingale_results['t_statistic']:.4f}")
            print(f"p值 = {martingale_results['p_value']:.4f}")
            print(f"均方误差 = {martingale_results['mse']:.4f}")
            return results
        else:
            self.count_all()
            results = self.regress()
            martingale_results = self.martingale_analysis()
            print("\n鞅分析结果：")
            print(f"t统计量 = {martingale_results['t_statistic']:.4f}")
            print(f"p值 = {martingale_results['p_value']:.4f}")
            print(f"均方误差 = {martingale_results['mse']:.4f}")
            return results

    def describe(self):
        """
        分别对W、P、S三人绘制每日分享次数的直方图。
        每日定义为当天7:00至次日4:00。
        """
        df = self.df.copy()
        df['time'] = pd.to_datetime(df['time'])
        def custom_day(ts):
            if ts.hour < 4:
                day = (ts - pd.Timedelta(days=1)).date()
            elif ts.hour < 7:
                day = (ts - pd.Timedelta(days=1)).date()
            else:
                day = ts.date()
            return day
        df['custom_day'] = df['time'].apply(custom_day)
        persons = ['W', 'P', 'S']
        for person in persons:
            daily_counts = df.groupby('custom_day')[person].sum()
            daily_counts = daily_counts.to_numpy() if hasattr(daily_counts, 'to_numpy') else np.array(daily_counts)
            daily_index = df.groupby('custom_day')[person].sum().index.astype(str)
            plt.figure(figsize=(10, 4))
            plt.bar(daily_index, daily_counts, color='skyblue')
            plt.xlabel('日期')
            plt.ylabel('分享次数')
            plt.title(f'{person}每日分享次数直方图')
            # 只显示分享次数最多的前3天的日期标签
            top_indices = np.argsort(daily_counts)[-3:]
            plt.xticks([daily_index[i] for i in top_indices], [str(daily_index[i]) for i in top_indices], rotation=45)
            plt.tight_layout()
            plt.savefig(f'{person}_daily_histogram.png')
            plt.show()

    def martingale_analysis(self, select=False, choose='P'):
        """
        分析分享计数过程是否满足鞅性质
        构造M(t)=N(t)-λt，检验其是否满足鞅性质
        """
        if select:
            self.select_data(choose)
            self.count_data()
            counts = self.back_df['count'].to_numpy()
            times = pd.to_datetime(self.back_df['time']).values
        else:
            self.count_all()
            counts = self.df['count'].to_numpy()
            times = pd.to_datetime(self.df['time']).values
        
        time_diffs = []
        for i in range(1, len(times)):
            current_time = pd.Timestamp(times[i])
            prev_time = pd.Timestamp(times[i-1])
            
            if current_time.hour >= 7:
                current_seven_am = current_time.replace(hour=7, minute=0, second=0, microsecond=0)
                if prev_time >= current_seven_am:
                    diff = (current_time - prev_time).total_seconds() / 60
                else:
                    diff = (current_time - current_seven_am).total_seconds() / 60
            else:
                if prev_time.date() == current_time.date() or prev_time.date() == current_time.date() - pd.Timedelta(days=1):
                    diff = (current_time - prev_time).total_seconds() / 60
                else:
                    prev_day_seven_am = (current_time - pd.Timedelta(days=1)).replace(hour=7, minute=0, second=0, microsecond=0)
                    diff = (current_time - prev_day_seven_am).total_seconds() / 60
            
            time_diffs.append(diff)
        
        time_diffs = np.array(time_diffs)
        cumulative_time = np.cumsum(time_diffs)
        
        lambda_hat = 1 / np.mean(time_diffs)
        
        # 构造鞅过程 M(t) = N(t) - λt
        N_t = counts[1:]  # 从第二个时间点开始的计数
        M_t = N_t - lambda_hat * cumulative_time
        
        increments = np.diff(M_t)

        # 计算增量的条件期望
        window_size = 15
        conditional_expectations = []
        for i in range(window_size, len(increments)):
            window = increments[i-window_size:i]
            conditional_expectations.append(np.mean(window))
        
        differences = increments[window_size:] - conditional_expectations
        
        # 进行鞅性检验
        # 检验增量的条件期望是否显著不为0
        t_stat, p_value = stats.ttest_1samp(differences, 0)
        
        mse = np.mean(differences**2)
        
        plt.figure(figsize=(15, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(cumulative_time, M_t, label='M(t)')
        plt.title('鞅过程 M(t) = N(t) - λt\np值 = {:.4f}, MSE = {:.4f}'.format(p_value, mse))
        plt.xlabel('时间（分钟）')
        plt.ylabel('M(t)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(increments, label='增量序列')
        plt.title('M(t)的增量序列')
        plt.xlabel('时间')
        plt.ylabel('增量')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        if select:
            plt.savefig(f'{self.back_name}_martingale.png')
        else:
            plt.savefig('Aggregate_martingale.png')
        plt.show()
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'mse': mse,
            'lambda_hat': lambda_hat
        }

def main():
    analysis = DataAnalysis('counts.csv')
    results = analysis.estimate()
    print("\n泊松过程拟合结果：")
    print(f"估计的到达率 λ = {results['lambda_hat']:.4f}")
    print(f"卡方统计量 = {results['chi2_statistic']:.4f}")
    print(f"p值 = {results['p_value']:.4f}")
    print(f"自相关检验结果：")
    print(results['lb_test'])
    
    # analysis.describe()

if __name__ == "__main__":
    main()

