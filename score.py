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
import json

pgf_with_latex = {"pgf.texsystem": 'pdflatex', "text.usetex": True, "font.family": "serif"}
matplotlib.rcParams.update(pgf_with_latex)

def create_accuracy_plot_and_return_mse(prediction_df, solution_array):
    prediction_array = prediction_df["Est Pump Difference, GPM"].to_numpy()
    plt.figure(figsize = (6,4))
    plt.plot(solution_array, prediction_array,'o',label = 'Estimates')
    plt.plot([-100,100],[-100,100],'--r',label = '1:1 line')
    plt.plot([solution_array[0], solution_array[0]],
             [prediction_array[0], solution_array[0]],
             '--',color = 'gray',label = 'misfit')
    for i in range(1,10):
        plt.plot([solution_array[i], solution_array[i]],
                 [prediction_array[i], solution_array[i]],
                 '--',color = 'gray')
    plt.xlabel('True, GPM'); 
    plt.ylabel('Prediction, GPM'); 
    plt.grid('on'); 
    plt.legend(); 
    plt.axis([-100,100,-100,100])
    plt.savefig('accuracy.pgf')
    return np.round(mean_squared_error(solution_array,prediction_array),3)

def create_realizations_plots(prediction_df, solution_array):
    prediction_realizations = prediction_df.iloc[:,2:].to_numpy()
    kde = [scipy.stats.gaussian_kde(prediction_realizations[i], 
                                    bw_method = None) for i in range(15)]
    t_range = [np.linspace(prediction_realizations[i].min() * 0.8,
                           prediction_realizations[i].max() * 1.2, 200) for i in range(15)]
    
    plt.figure(figsize =(8,10))
    for i, item in enumerate([4, 31, 42, 52, 71, 76, 96, 131, 137, 194, 220, 236, 265, 321, 345]):
        ax = plt.subplot(8,2,i+1)
        if i == 0:
            pdf, = ax.plot(t_range[i],kde[i](t_range[i]), lw=2, label=f'PDF of reals')
            real, = ax.plot(solution_array[i], 0,'ro',markersize = 10, label ='True')
            plt.xlabel('Prediction, MSTB');
            plt.title(f'Well ID {item}')
        else:
            pdf, = ax.plot(t_range[i],kde[i](t_range[i]), lw=2)
            real, = ax.plot(solution_array[i], 0, 'ro',markersize = 10)
            plt.title(f'Well ID {item}')
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
            # print(solution_array[j])
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
    #plt.fill([i for i in range(11)],goodness_score,alpha = 0.2, label = 'misfit area')
    plt.xticks([i for i in np.linspace(0,10,6)], [f'{np.int64(i*10)}%' for i in np.linspace(0,10,6)])
    plt.yticks([i for i in np.linspace(0,10,6)], [f'{np.int64(i*10)}%' for i in np.linspace(0,10,6)])
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

    return np.abs(np.round(goodness_score_,3))

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

def create_team_report(team_name, mse, 
                       goodness_score, 
                       presentation_comments,
                       code_review_comments):

    template = latex_jinja_env.get_template('report_template.tex')

    #Render tex and write to file
    rendered_tex = template.render(teamname=team_name.replace('_', ' '), 
                                   mse=mse,
                                   goodness=goodness_score,
                                   presentationComments=presentation_comments,
                                   codereviewComments=code_review_comments)
    tex_filename = f'{team_name}_report.tex'
    with open(tex_filename, 'w') as f:
        f.write(rendered_tex)

    #run latex on the file
    subprocess.run(['latexmk', '-pdf', 
                    '-output-directory=reports', tex_filename])
    return

def parse_team_name(team_name):
    return team_name.replace(' ', '').replace('-', '').replace('_', '').lower()

def parse_question_scores(score):
    return int(score.split(',')[0])

def get_presentation_dataframes(presentation_csv):

    df = pd.read_csv(presentation_csv)
    df = df.drop([0,1])
    df = df.astype('str')
    df['Q1'] = df['Q1'].apply(parse_team_name)
    df['Q2'] = df['Q2'].apply(parse_question_scores)
    df['Q3'] = df['Q3'].apply(parse_question_scores)
    df['Q4'] = df['Q4'].apply(parse_question_scores)
    df['Q5'] = df['Q5'].apply(parse_question_scores)
    df['Q6'] = df['Q6'].apply(parse_question_scores)
    df['Q7'] = df['Q7'].apply(parse_question_scores)
    df['Q8'] = df['Q8'].apply(parse_question_scores)

    scores_df = df.loc[:, 'Q1':'Q8'].groupby('Q1').mean().sum(axis=1)
    comments_df = df[['Q1','Q9']].groupby('Q1')['Q9'].apply(list)

    return scores_df, comments_df

def get_code_review_dataframes(code_review_csv):

    df = pd.read_csv(code_review_csv)
    df = df.drop([0,1])
    df = df.astype('str')
    df['Q1'] = df['Q1'].apply(parse_team_name)
    df['Q2'] = df['Q2'].apply(parse_question_scores)
    df['Q3'] = df['Q3'].apply(parse_question_scores)
    df['Q4'] = df['Q4'].apply(parse_question_scores)
    df['Q5'] = df['Q5'].apply(parse_question_scores)
    df['Q6'] = df['Q6'].apply(parse_question_scores)

    scores_df = df.loc[:, 'Q1':'Q6'].groupby('Q1').mean().sum(axis=1)
    comments_df = df[['Q1','Q7']].groupby('Q1')['Q7'].apply(list)

    return scores_df, comments_df

# %%

if __name__ == '__main__':

    from github import Github
    from io import StringIO


    gh = Github(os.environ['PGEHACKATHON_SECRET_TOKEN'])

    repos = gh.get_organization_repos('PGEHackathon')

    blocked_list = ['PGEHackathon/data', 'PGEHackathon/workshop', 
                    'PGEHackathon/scoring', 'PGEHackathon/PGEHackathon', 
                    'PGEHackathon/resources', 'PGEHackathon/TheNomads', 
                    'PGEHackathon/hidden', 'PGEHackathon/data2021', 'PGEHackathon/data2022'
                    'PGEHackathon/data2023']
    repos = ['PGEHackathon/NoFreeLunch']

    # Get answers
    result = gh.get_file_in_repo('answer.csv', 'PGEHackathon/hidden')
    solution_array = pd.read_csv(StringIO(result)).iloc[:, 1].to_numpy()


    team_names = []
    team_mse = []
    team_goodness_score = []
    for repo in repos:


        if repo not in blocked_list:
            print(f"Generating Report For: {repo}")

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

                presentation_score_df, presentation_comments_df = \
                    get_presentation_dataframes('presentation.csv')

                code_review_score_df, code_review_comments_df = \
                    get_code_review_dataframes('code_review.csv')


                try:
                    presentation_comments = \
                        presentation_comments_df[parse_team_name(team_name)]
                    code_review_comments = \
                        code_review_comments_df[parse_team_name(team_name)]
                except:
                    presentation_comments = ["None"]
                    code_review_comments = ["None"]


                create_team_report(team_name, mse, 
                                   goodness_score, 
                                   presentation_comments,
                                   code_review_comments)

    df = pd.DataFrame(np.array([team_names, 
                                team_mse, 
                                team_goodness_score]).T, columns=['Team Names',
                                                                  'MSE',
                                                                  'Goodness Score'])
    df['Short Names'] = df['Team Names'].apply(parse_team_name)
    df.set_index(['Short Names'], inplace=True)

    df['Pres. Score'] = presentation_score_df
    df['Code Score'] = code_review_score_df
    df['MSE Rank'] = df['MSE'].astype('float64').rank(method='min', ascending=True, na_option='top')
    df['Goodness Rank'] = df['Goodness Score'].astype('float64').rank(method='min', ascending=False, na_option='top')
    df['Pres. Rank'] = df['Pres. Score'].astype('float64').rank(method='min', ascending=False, na_option='top')
    df['Code Rank'] = df['Code Score'].astype('float64').rank(method='min', ascending=False, na_option='top')

    df['Overall Rank'] = (0.375 * df['MSE Rank'].astype('float64') + 
                          0.375 * df['Goodness Rank'].astype('float64') + 
                          0.200 * df['Pres. Rank'].astype('float64') + 
                          0.050 * df['Code Rank'].astype('float64')
                         ).rank(method='min')

    df.sort_values('Overall Rank', inplace=True)
    df.index.name = None
    #df.reset_index(inplace=True)
    with open('final_rankings_table.tex', 'w') as f:
        df.to_latex(index=False, buf=f)

    subprocess.run(['latexmk', '-pdf', 
                    '-output-directory=reports', 'final_report.tex'])
