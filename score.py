#!/usr/bin/env python

import scipy.stats
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jinja2
from jinja2 import Template
import os
import subprocess
import matplotlib

pgf_with_latex = {"pgf.texsystem": 'pdflatex'}
matplotlib.rcParams.update(pgf_with_latex)

def create_accuracy_plot_and_return_mse(prediction_df, solution_array):
    prediction_array = prediction_df['Prediction, MSTB'].to_numpy()
    plt.plot(solution_array, prediction_array,'o',label = 'Estimates')
    plt.plot([500,2000],[500,2000],'--r',label = '1:1 line')
    plt.plot([solution_array[0], solution_array[0]],
             [prediction_array[0], solution_array[0]],
             '--',color = 'gray',label = 'misfit')
    for i in range(1,10):
        plt.plot([solution_array[i], solution_array[i]],
                 [prediction_array[i], solution_array[i]],
                 '--',color = 'gray')
    plt.xlabel('True, MSTB'); 
    plt.ylabel('Prediction, MSTB'); 
    plt.grid('on'); 
    plt.legend(); 
    plt.axis([500,2000,500,2000])
    plt.savefig('accuracy.pgf')
    return np.round(mean_squared_error(solution_array,prediction_array),3)

def create_realizations_plots(prediction_df, solution_array):
    prediction_realizations = prediction_df.iloc[:,2:].to_numpy()
    kde = [scipy.stats.gaussian_kde(prediction_realizations[i], 
                                    bw_method = None) for i in range(10)]
    t_range = [np.linspace(prediction_realizations[i].min() * 0.8,
                           prediction_realizations[i].max() * 1.2, 200) for i in range(10)]
    
    plt.figure(figsize =(6,8))
    for i in range(10):
        ax = plt.subplot(5,2,i+1)
        if i == 0:
            pdf, = ax.plot(t_range[i],kde[i](t_range[i]), lw=2, label=f'PDF of reals')
            real, = ax.plot(solution_array[i], 0,'ro',markersize = 10, label ='True')
            plt.xlabel('Prediction, MSTB');
            plt.title(f'Preproduction No. {74+i}')
        else:
            pdf, = ax.plot(t_range[i],kde[i](t_range[i]), lw=2)
            real, = ax.plot(solution_array[i], 0, 'ro',markersize = 10)
            plt.title(f'Preproduction No. {74+i}')
        if i == 1:
            plt.legend([pdf, real], ['PDF of realz.', 'True'], bbox_to_anchor=(1.05, 1.0), loc='upper left')
        if i % 2 == 0:
            plt.ylabel('Probability')

    plt.tight_layout()
    plt.savefig('realizations.pgf')
    return

def compute_goodness_array(prediction_df, solution_array):
    prediction_realizations = prediction_df.iloc[:,2:].to_numpy()
    goodness_score = []
    list_percentile_lower = [50-5*i for i in range(0,11)]   # Define upper/lower boundary of "within-percentile" ranges
    list_percentile_upper = [50+5*i for i in range(0,11)]   # E.g., 10% range - from 45% to 55% percentile

    for i in range(11):     # 0%, 10%, 20%, 30%, ... 100% percentiles ranges
        num_within = 0      # Counts for wells within the range
        for j in range(10): # 10 Predrill wells
            min_ = np.percentile(prediction_realizations[j],list_percentile_lower[i])
            max_ = np.percentile(prediction_realizations[j],list_percentile_upper[i])
            if solution_array[j] > min_ and solution_array[j] < max_:
                num_within += 1
        goodness_score.append(num_within)
    return goodness_score

def create_goodness_plot_and_return_goodness_score(prediction_df, solution_array):
    goodness_score = compute_goodness_array(prediction_df, solution_array)
    prediction_realizations = prediction_df.iloc[:,2:].to_numpy()

    plt.figure(figsize = (6,4))
    plt.plot(goodness_score,'--ko', label = 'Goodness')
    plt.plot([0,10],[0,10], '-r', label = '1:1 line')
    plt.fill([i for i in range(11)],goodness_score,alpha = 0.2, label = 'misfit area')
    plt.xticks([i for i in np.linspace(0,10,6)], [f'{np.int(i*10)}%' for i in np.linspace(0,10,6)])
    plt.yticks([i for i in np.linspace(0,10,6)], [f'{np.int(i*10)}%' for i in np.linspace(0,10,6)])
    plt.xlabel('Within percentile'); 
    plt.ylabel('Percentage of wells within the range')
    plt.legend(); 
    plt.savefig('goodness.pgf')

    ## Total area of plot is 100 (square of 10 x 10)
    # If goodness plot perfectly along with 1:1 line, that should be "1" 
    # If goodness plot all flat at 0% in y-axis, that should be "0" 
    # If goodness plot all flat at 100% in y-axis, that should be "0.5" 
    # Follow lines of code compute and normalize area above/below area to get goodness score (0~1)
    goodness_score_upNdown = np.array(goodness_score) - np.arange(0,11) 
    a_interval_index = [1 if goodness_score[i+1] >= i+1 else 0 for i in range(10)]
    goodness_score_ = 1
    for i in range(10):
        if a_interval_index[i] == 1:
            goodness_score_ -= +1/2*goodness_score_upNdown[i+1]/45
        else:
            goodness_score_ -= -goodness_score_upNdown[i+1]/55

    return np.round(goodness_score_,3)

latex_jinja_env = jinja2.Environment(
    block_start_string = '\BLOCK{',
    block_end_string = '}',
    variable_start_string = '\VAR{',
    variable_end_string = '}',
        comment_start_string = '\#{',
        comment_end_string = '}',
        line_statement_prefix = '%%',
        line_comment_prefix = '%#',
        trim_blocks = True,
        autoescape = False,
        loader = jinja2.FileSystemLoader(os.path.abspath('.'))
)

def create_team_report(team_name, mse, goodness_score):

    template = latex_jinja_env.get_template('report_template.tex')

    #Render tex and write to file
    rendered_tex = template.render(teamname=team_name, 
                                   mse=mse,
                                   goodness=goodness_score)
    tex_filename = f'{team_name}_report.tex'
    with open(tex_filename, 'w') as f:
        f.write(rendered_tex)

    #run latex on the file
    subprocess.run(['latexmk', '-pdf', 
                    '-output-directory=reports', tex_filename])
    return



if __name__ == '__main__':

    from github import Github
    from io import StringIO


    gh = Github(os.environ['PGEHACKATHON_SECRET_TOKEN'])

    repos = gh.get_organization_repos('PGEHackathon')

    blocked_list = ['PGEHackathon/data', 'PGEHackathon/workshop', 
                    'PGEHackathon/scoring', 'PGEHackathon/PGEHackathon', 
                    'PGEHackathon/resources']

    solution_array = np.load('True_for_predrill_3yr.npy') # Solution

    team_names = []
    team_mse = []
    team_goodness_score = []
    for repo in repos:

        if repo not in blocked_list:

            result = gh.get_file_in_repo('solution.csv', repo)

            if result is not None:

                team_name = repo.split('/')[1]
                team_names.append(team_name)

                prediction_df = pd.read_csv(StringIO(result))

                mse = create_accuracy_plot_and_return_mse(prediction_df, 
                                                          solution_array)
                team_mse.append(mse)

                create_realizations_plots(prediction_df, solution_array)

                goodness_score = \
                    create_goodness_plot_and_return_goodness_score(prediction_df,
                                                                   solution_array)
                team_goodness_score.append(goodness_score)

                create_team_report(team_name, mse, goodness_score)

    df = pd.DataFrame(np.array([team_names, 
                                team_mse, 
                                team_goodness_score]).T, columns=['Team Names',
                                                                  'MSE',
                                                                  'Goodness Score'])
    df['MSE Rank'] = df['MSE'].astype('float64').rank(method='min', ascending=True).astype('int')
    df['Goodness Score Rank'] = df['Goodness Score'].astype('float64').rank(method='min', ascending=False).astype('int')
    df['Overall Rank'] = (df['MSE Rank'].astype('float64') + df['Goodness Score Rank'].astype('float64')).rank(method='min').astype('int')

    df.sort_values('Overall Rank', inplace=True)

    with open('final_rankings_table.tex', 'w') as f:
        df.to_latex(index=False, buf=f)

    subprocess.run(['latexmk', '-pdf', 
                    '-output-directory=reports', 'final_report.tex'])
