import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.colors as mcolors
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import LinearSegmentedColormap

class WelfareCliffVisualizer:
    """
    福利悬崖可视化类
    
    生成静态图表、交互式图表和动态可视化
    """
    
    def __init__(self, output_dir: str = "../output"):
        """
        初始化可视化器
        
        参数:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        self.figures_dir = os.path.join(output_dir, "figures")
        self.html_dir = os.path.join(output_dir, "html")
        
        # 确保输出目录存在
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.html_dir, exist_ok=True)
        
        # 定义颜色方案
        self.colors = {
            'income': '#fdae61',
            'benefits': '#2c7bb6',
            'cliff': '#d7191c',
            'net': '#1a9641',
            'snap': '#4575b4',
            'housing': '#74add1',
            'medicaid': '#abd9e9',
            'tanf': '#e0f3f8'
        }
        
        # 设置可视化默认样式
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 10
        
    def plot_income_benefit_curve(self, 
                               df: pd.DataFrame, 
                               title: str = "收入-福利曲线", 
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制收入-福利曲线
        
        参数:
            df: 包含不同收入水平下福利和净收入的DataFrame
            title: 图表标题
            save_path: 保存路径，如果为None则不保存
            
        返回:
            Matplotlib Figure对象
        """
        fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
        
        # 绘制总福利曲线
        ax1.plot(df['gross_income'], df['total_benefits'], 
                color=self.colors['benefits'], 
                linewidth=2,
                label='总福利')
        ax1.set_xlabel('税前收入')
        ax1.set_ylabel('福利金额')
        ax1.tick_params(axis='y')
        
        # 创建第二个y轴
        ax2 = ax1.twinx()
        
        # 绘制边际税率曲线
        ax2.plot(df['gross_income'], df['marginal_tax_rate'], 
                color=self.colors['cliff'], 
                linewidth=2, 
                linestyle='--',
                label='边际税率')
        ax2.set_ylabel('边际税率')
        ax2.set_ylim(0, 1.05)
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax2.tick_params(axis='y')
        
        # 添加水平线标记50%边际税率
        ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
        
        # 组合图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.figures_dir, save_path), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_benefit_composition(self, 
                               df: pd.DataFrame, 
                               title: str = "福利构成分析", 
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制福利构成分析堆叠面积图
        
        参数:
            df: 包含不同收入水平下福利和净收入的DataFrame
            title: 图表标题
            save_path: 保存路径，如果为None则不保存
            
        返回:
            Matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        # 创建堆叠面积图
        ax.fill_between(df['gross_income'], 0, df['tanf'], 
                        color=self.colors['tanf'], alpha=0.8, label='TANF')
        ax.fill_between(df['gross_income'], df['tanf'], df['tanf'] + df['snap'], 
                        color=self.colors['snap'], alpha=0.8, label='SNAP')
        ax.fill_between(df['gross_income'], df['tanf'] + df['snap'], 
                        df['tanf'] + df['snap'] + df['housing'], 
                        color=self.colors['housing'], alpha=0.8, label='住房补贴')
        ax.fill_between(df['gross_income'], df['tanf'] + df['snap'] + df['housing'],
                        df['total_benefits'], 
                        color=self.colors['medicaid'], alpha=0.8, label='医疗补助')
        
        # 添加净收入曲线
        ax.plot(df['gross_income'], df['net_income'], 
                color=self.colors['net'], 
                linewidth=2,
                label='净收入')
        
        # 在净收入曲线下添加直线
        ax.plot(df['gross_income'], df['gross_income'], 
                color='black', 
                linewidth=1,
                linestyle='--',
                label='税前收入')
        
        ax.set_xlabel('税前收入')
        ax.set_ylabel('金额')
        ax.legend(loc='upper left')
        
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.figures_dir, save_path), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_state_comparison(self, 
                            df: pd.DataFrame, 
                            top_n: int = 10,
                            metric: str = 'total_benefits',
                            title: str = "各州福利比较", 
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制各州福利比较条形图
        
        参数:
            df: 包含不同州福利信息的DataFrame
            top_n: 显示前N个州
            metric: 比较指标，可选'total_benefits'或'marginal_tax_rate'等
            title: 图表标题
            save_path: 保存路径，如果为None则不保存
            
        返回:
            Matplotlib Figure对象
        """
        # 按照指定指标降序排列
        sorted_df = df.sort_values(by=metric, ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        # 创建条形图
        bars = ax.bar(sorted_df['state'], sorted_df[metric], 
                    color=self.colors['benefits'])
        
        ax.set_xlabel('州')
        
        if metric == 'total_benefits':
            ax.set_ylabel('总福利金额')
        elif metric == 'marginal_tax_rate':
            ax.set_ylabel('边际税率')
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        else:
            ax.set_ylabel(metric)
        
        # 在条形上添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', rotation=0)
        
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.figures_dir, save_path), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_historical_trends(self, 
                             df: pd.DataFrame, 
                             metrics: List[str] = ['total_benefits', 'medicaid'],
                             title: str = "福利历史趋势", 
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制福利历史趋势折线图
        
        参数:
            df: 包含不同年份福利信息的DataFrame
            metrics: 要绘制的指标列表
            title: 图表标题
            save_path: 保存路径，如果为None则不保存
            
        返回:
            Matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        # 绘制每个指标的历史趋势
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                ax.plot(df['year'], df[metric], 
                        marker='o', 
                        linewidth=2,
                        label=metric)
        
        # 标记重要政策变化点
        policy_years = {
            1996: '福利改革法案',
            2010: '平价医疗法案',
            2014: 'ACA医疗补助扩展'
        }
        
        for year, policy in policy_years.items():
            if year in df['year'].values:
                ax.axvline(x=year, color='gray', linestyle=':', alpha=0.7)
                ax.text(year, ax.get_ylim()[1]*0.9, policy, 
                        rotation=90, ha='right', va='top')
        
        ax.set_xlabel('年份')
        ax.set_ylabel('福利金额')
        ax.legend()
        
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.figures_dir, save_path), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_policy_scenario_comparison(self, 
                                      scenario_results: Dict[str, pd.DataFrame],
                                      title: str = "政策情景比较", 
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制政策情景比较图
        
        参数:
            scenario_results: 包含基准和情景模拟结果的字典
            title: 图表标题
            save_path: 保存路径，如果为None则不保存
            
        返回:
            Matplotlib Figure对象
        """
        base_df = scenario_results['base']
        scenario_df = scenario_results['scenario']
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        # 绘制基准和情景的总福利曲线
        ax.plot(base_df['gross_income'], base_df['total_benefits'], 
                color='blue', 
                linewidth=2,
                label='基准情景')
        
        ax.plot(scenario_df['gross_income'], scenario_df['total_benefits'], 
                color='red', 
                linewidth=2,
                label='政策情景')
        
        # 填充两条曲线之间的区域
        ax.fill_between(base_df['gross_income'], 
                        base_df['total_benefits'], 
                        scenario_df['total_benefits'],
                        where=scenario_df['total_benefits'] >= base_df['total_benefits'],
                        color='green', alpha=0.3, label='福利增加')
        
        ax.fill_between(base_df['gross_income'], 
                        base_df['total_benefits'], 
                        scenario_df['total_benefits'],
                        where=scenario_df['total_benefits'] < base_df['total_benefits'],
                        color='red', alpha=0.3, label='福利减少')
        
        ax.set_xlabel('税前收入')
        ax.set_ylabel('总福利金额')
        ax.legend()
        
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.figures_dir, save_path), dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_benefit_cliff(self, 
                                       df: pd.DataFrame,
                                       title: str = "交互式福利悬崖分析",
                                       save_path: Optional[str] = None) -> go.Figure:
        """
        创建交互式福利悬崖分析图
        
        参数:
            df: 包含不同收入水平下福利和净收入的DataFrame
            title: 图表标题
            save_path: 保存路径，如果为None则不保存
            
        返回:
            Plotly Figure对象
        """
        # 创建子图
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 添加总福利曲线
        fig.add_trace(
            go.Scatter(
                x=df['gross_income'], 
                y=df['total_benefits'],
                mode='lines',
                name='总福利',
                line=dict(color=self.colors['benefits'], width=3)
            )
        )
        
        # 添加各项福利
        fig.add_trace(
            go.Scatter(
                x=df['gross_income'], 
                y=df['snap'],
                mode='lines',
                name='SNAP',
                line=dict(color=self.colors['snap'], width=2)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['gross_income'], 
                y=df['housing'],
                mode='lines',
                name='住房补贴',
                line=dict(color=self.colors['housing'], width=2)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['gross_income'], 
                y=df['medicaid'],
                mode='lines',
                name='医疗补助',
                line=dict(color=self.colors['medicaid'], width=2)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['gross_income'], 
                y=df['tanf'],
                mode='lines',
                name='TANF',
                line=dict(color=self.colors['tanf'], width=2)
            )
        )
        
        # 添加净收入曲线
        fig.add_trace(
            go.Scatter(
                x=df['gross_income'], 
                y=df['net_income'],
                mode='lines',
                name='净收入',
                line=dict(color=self.colors['net'], width=3)
            )
        )
        
        # 添加税前收入参考线
        fig.add_trace(
            go.Scatter(
                x=df['gross_income'], 
                y=df['gross_income'],
                mode='lines',
                name='税前收入',
                line=dict(color='black', width=2, dash='dash')
            )
        )
        
        # 在第二个y轴添加边际税率
        fig.add_trace(
            go.Scatter(
                x=df['gross_income'], 
                y=df['marginal_tax_rate'],
                mode='lines',
                name='边际税率',
                line=dict(color=self.colors['cliff'], width=3, dash='dash')
            ),
            secondary_y=True
        )
        
        # 更新布局
        fig.update_layout(
            title=title,
            xaxis_title='税前收入',
            yaxis_title='金额',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(
                family="Arial",
                size=12
            ),
            hovermode="x unified"
        )
        
        # 更新第二个y轴
        fig.update_yaxes(
            title_text="边际税率", 
            range=[0, 1.05], 
            tickformat=".0%",
            secondary_y=True
        )
        
        # 添加水平参考线表示50%边际税率
        fig.add_shape(
            type="line",
            x0=min(df['gross_income']),
            y0=0.5,
            x1=max(df['gross_income']),
            y1=0.5,
            line=dict(color="gray", width=1, dash="dot"),
            secondary_y=True
        )
        
        # 找出边际税率超过50%的点
        cliff_points = df[df['marginal_tax_rate'] >= 0.5]
        
        # 添加标记悬崖点
        if not cliff_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=cliff_points['gross_income'],
                    y=cliff_points['marginal_tax_rate'],
                    mode='markers',
                    marker=dict(
                        color=self.colors['cliff'],
                        size=10,
                        symbol='circle'
                    ),
                    name='悬崖点',
                    hovertemplate='收入: %{x}<br>边际税率: %{y:.1%}<extra></extra>'
                ),
                secondary_y=True
            )
        
        if save_path:
            fig.write_html(os.path.join(self.html_dir, save_path))
        
        return fig
    
    def create_interactive_state_comparison(self, 
                                         df: pd.DataFrame,
                                         year: int,
                                         title: str = "州际福利差异分析",
                                         save_path: Optional[str] = None) -> go.Figure:
        """
        创建交互式州际福利差异分析图
        
        参数:
            df: 包含不同州福利信息的DataFrame
            year: 年份
            title: 图表标题
            save_path: 保存路径，如果为None则不保存
            
        返回:
            Plotly Figure对象
        """
        # 创建地图图表
        fig = px.choropleth(
            df,
            locations='state',
            locationmode='USA-states',
            color='total_benefits',
            color_continuous_scale=px.colors.diverging.RdBu,
            scope="usa",
            labels={'total_benefits': '总福利金额'},
            title=f"{title} ({year}年)"
        )
        
        # 更新布局
        fig.update_layout(
            font=dict(
                family="Arial",
                size=12
            ),
            coloraxis_colorbar=dict(
                title='福利金额'
            )
        )
        
        # 添加悬停信息
        fig.update_traces(
            hovertemplate='<b>%{location}</b><br>总福利: %{z:,.0f}<br><extra></extra>'
        )
        
        if save_path:
            fig.write_html(os.path.join(self.html_dir, save_path))
        
        return fig
    
    def create_animated_historical_map(self, 
                                    dfs: Dict[int, pd.DataFrame],
                                    metric: str = 'total_benefits',
                                    title: str = "福利变化历史动画",
                                    save_path: Optional[str] = None) -> None:
        """
        创建历史变化动画地图
        
        参数:
            dfs: 包含不同年份DataFrame的字典，键为年份
            metric: 要可视化的指标
            title: 图表标题
            save_path: 保存路径，如果为None则不保存
        """
        # 准备动画帧
        frames = []
        years = sorted(dfs.keys())
        
        # 找出所有年份中指标的最大值和最小值
        all_values = []
        for year in years:
            all_values.extend(dfs[year][metric].tolist())
        
        vmin = min(all_values)
        vmax = max(all_values)
        
        # 创建颜色映射
        cmap = LinearSegmentedColormap.from_list("custom_cmap", 
                                                [self.colors['benefits'], 
                                                self.colors['cliff']])
        
        # 创建基础图形
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        
        def update(frame):
            year = years[frame]
            ax.clear()
            
            current_df = dfs[year]
            
            # 创建地图
            for state in current_df['state']:
                value = current_df[current_df['state'] == state][metric].values[0]
                # 这里需要实际的地图绘制逻辑
                # 简化的示例使用文本代替
                ax.text(0.5, 0.5, f"年份: {year}\n示例地图")
            
            ax.set_title(f"{title} - {year}年")
            
        ani = animation.FuncAnimation(fig, update, frames=len(years), interval=1000)
        
        if save_path:
            ani.save(os.path.join(self.figures_dir, save_path), writer='pillow', fps=1)
        
        plt.close(fig)
    
    def create_interactive_policy_impact(self, 
                                      expansion_df: pd.DataFrame,
                                      non_expansion_df: pd.DataFrame,
                                      title: str = "医疗补助扩展影响分析",
                                      save_path: Optional[str] = None) -> go.Figure:
        """
        创建交互式政策影响分析图
        
        参数:
            expansion_df: 扩展州DataFrame
            non_expansion_df: 非扩展州DataFrame
            title: 图表标题
            save_path: 保存路径，如果为None则不保存
            
        返回:
            Plotly Figure对象
        """
        fig = go.Figure()
        
        # 添加扩展州的净收入曲线
        fig.add_trace(
            go.Scatter(
                x=expansion_df['gross_income'],
                y=expansion_df['net_income'],
                mode='lines',
                name='扩展州净收入',
                line=dict(color='blue', width=3)
            )
        )
        
        # 添加非扩展州的净收入曲线
        fig.add_trace(
            go.Scatter(
                x=non_expansion_df['gross_income'],
                y=non_expansion_df['net_income'],
                mode='lines',
                name='非扩展州净收入',
                line=dict(color='red', width=3)
            )
        )
        
        # 添加扩展州的医疗补助
        fig.add_trace(
            go.Scatter(
                x=expansion_df['gross_income'],
                y=expansion_df['medicaid'],
                mode='lines',
                name='扩展州医疗补助',
                line=dict(color='blue', width=2, dash='dash')
            )
        )
        
        # 添加非扩展州的医疗补助
        fig.add_trace(
            go.Scatter(
                x=non_expansion_df['gross_income'],
                y=non_expansion_df['medicaid'],
                mode='lines',
                name='非扩展州医疗补助',
                line=dict(color='red', width=2, dash='dash')
            )
        )
        
        # 计算差异并添加
        diff_income = expansion_df['gross_income'].copy()
        diff_net = expansion_df['net_income'] - non_expansion_df['net_income']
        
        fig.add_trace(
            go.Scatter(
                x=diff_income,
                y=diff_net,
                mode='lines',
                name='净收入差异',
                line=dict(color='green', width=2),
                visible='legendonly'  # 默认隐藏，可通过图例切换
            )
        )
        
        # 添加参考线
        fig.add_trace(
            go.Scatter(
                x=expansion_df['gross_income'],
                y=expansion_df['gross_income'],
                mode='lines',
                name='税前收入',
                line=dict(color='black', width=1, dash='dot')
            )
        )
        
        # 更新布局
        fig.update_layout(
            title=title,
            xaxis_title='税前收入',
            yaxis_title='金额',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(
                family="Arial",
                size=12
            ),
            hovermode="x unified"
        )
        
        # 添加注释，标记扩展政策影响最显著的区域
        significant_diff_idx = diff_net.idxmax()
        significant_income = diff_income.iloc[significant_diff_idx]
        significant_diff = diff_net.iloc[significant_diff_idx]
        
        fig.add_annotation(
            x=significant_income,
            y=expansion_df['net_income'].iloc[significant_diff_idx],
            text="政策影响<br>最显著点",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
        
        if save_path:
            fig.write_html(os.path.join(self.html_dir, save_path))
        
        return fig 