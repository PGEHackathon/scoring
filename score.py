#!/usr/bin/env python

import scipy.stats
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jinja2
from jinja2 import Template
import os
import subprocess
import UTuning.plots
import UTuning.scorer

plt.rcParams.update({
    "pgf.texsystem": "lualatex",        # Change to your LaTeX system
})
#matplotlib.rcParams.update(pgf_with_latex)

def create_accuracy_plot_and_return_mape(prediction_df, solution_array):

    prediction_df = prediction_df.set_index(['Masked Well Name'])

    # Initialize empty column for normalized values
    prediction_df['Normalized Fuel Value'] = 0.0
    normalized_solution_array = np.zeros(len(prediction_df), dtype=np.double)

    # Normalize each fuel type separately
    for fuel_type in ['Grid', 'Turbine', 'Diesel', 'DBG_CNG', 'DBG_Diesel']:
        mask = prediction_df['Fuel Type'] == fuel_type
        if mask.any():
            #Normalize prediction
            fuel_values = prediction_df.loc[mask, 'Fuel Value']
            min_val = fuel_values.min()
            max_val = fuel_values.max()
            if max_val > min_val:  # Avoid division by zero
                prediction_df.loc[mask, 'Normalized Fuel Value'] = (fuel_values - min_val) / (max_val - min_val)

            #Normalize solution array
            fuel_values = solution_array[mask]
            min_val = np.min(fuel_values)
            max_val = np.max(fuel_values)
            if max_val > min_val:  # Avoid division by zero
                normalized_solution_array[mask] = (fuel_values - min_val) / (max_val - min_val)


    normalized_prediction_array = prediction_df["Normalized Fuel Value"].to_numpy()

    plt.figure(figsize = (6,4))
    plt.plot(normalized_solution_array, normalized_prediction_array,'o',label = 'Estimates')
    plt.plot([-0.05,1.05],[-0.05,1.05],'--r',label = '1:1 line')
    plt.plot([normalized_solution_array[0], normalized_solution_array[0]],
             [normalized_prediction_array[0], normalized_solution_array[0]],
             '--',color = 'gray',label = 'misfit')
    for i in range(1,len(solution_array)):
        plt.plot([normalized_solution_array[i], normalized_solution_array[i]],
                 [normalized_prediction_array[i], normalized_solution_array[i]],
                 '--',color = 'gray')
    plt.xlabel('Normalized True Value'); 
    plt.ylabel('Normalized Prediction'); 
    plt.grid('on'); 
    plt.legend(); 
    plt.axis([-0.05,1.05,-0.05,1.05])
    # plt.show()
    plt.savefig('accuracy.pgf')
    return np.round(mean_absolute_percentage_error(solution_array,prediction_df['Fuel Value'].to_numpy()),3)

def create_realizations_plots(prediction_df, solution_array):
    prediction_df = prediction_df.head(10)
    well_names = prediction_df['Masked Well Name'].tolist()
    fuel_type = prediction_df['Fuel Type'].tolist()
    prediction_realizations = prediction_df.iloc[:,3:].to_numpy()
    kde = [scipy.stats.gaussian_kde(prediction_realizations[i], 
                                    bw_method = None) for i in range(10)]
    t_range = [np.linspace(prediction_realizations[i].min() * 0.8,
                           prediction_realizations[i].max() * 1.2, 200) for i in range(10)]
    
    plt.figure(figsize =(8,10))
    for i, name in enumerate(well_names):
        ax = plt.subplot(5,2,i+1)
        if i == 0:
            pdf, = ax.plot(t_range[i],kde[i](t_range[i]), lw=2, label=f'PDF of reals')
            real, = ax.plot(solution_array[i], 0,'ro',markersize = 10, label ='True')
            plt.xlabel('Prediction');
            plt.title(f'Well: {name} \nFuel Type: {fuel_type[i]}')
        else:
            pdf, = ax.plot(t_range[i],kde[i](t_range[i]), lw=2)
            real, = ax.plot(solution_array[i], 0, 'ro',markersize = 10)
            plt.title(f'Well ID {name} \nFuel Type: {fuel_type[i]}')
        if i == 1:
            plt.legend([pdf, real], ['PDF of realz.', 'True'], bbox_to_anchor=(1.05, 1.0), loc='upper left')
        if i % 2 == 0:
            plt.ylabel('Probability')

    plt.tight_layout()
    plt.savefig('realizations.pgf')
    return

def create_goodness_plot_and_return_goodness_score(prediction_df, solution_array):

    prediction_realizations = prediction_df.iloc[:,3:].to_numpy()
    std_array = np.std(prediction_realizations, axis=1)

    score = UTuning.scorer.scorer(prediction_realizations, solution_array, std_array)

    IF_array = score.IndicatorFunction()

    n_quantiles = 11
    perc = np.linspace(0.0, 1.00, n_quantiles)

    fig, ax = UTuning.plots.goodness_plot(perc, IF_array, prediction_realizations, solution_array, std_array)
    ax.legend(loc='upper left', bbox_to_anchor=(0,1))
    fig.savefig('goodness.pgf')
    plt.close()

    return np.round(score.Goodness(), 3)




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

def create_team_report(team_name, mape, 
                       goodness_score, 
                       presentation_comments,
                       code_review_comments):

    template = latex_jinja_env.get_template('report_template.tex')

    #Render tex and write to file
    rendered_tex = template.render(teamname=team_name.replace('_', ' '), 
                                   mape=mape,
                                   goodness=goodness_score,
                                   presentationComments=presentation_comments,
                                   codereviewComments=code_review_comments)
    tex_filename = f'{team_name}_report.tex'
    with open(tex_filename, 'w') as f:
        f.write(rendered_tex)

    #run latex on the file
    subprocess.run(['latexmk', '-pdflua', 
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
    df['Q6'] = df['Q6'].apply(parse_question_scores) * 3
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
                    'PGEHackathon/resources',  
                    'PGEHackathon/hidden', 'PGEHackathon/data2021', 'PGEHackathon/data2022'
                    'PGEHackathon/data2023', 'PGEHackathon/data2024', 
                    'PGEHackathon/pge-hackathon2025-GuabaPy', 
                    'PGEHackathon/EulersOilers', 
                    'PGEHackathon/PartyRockers', 
                    'PGEHackathon/rubber_duckies', 
                    'PGEHackathon/VOX']

    # Get answers
    result = gh.get_file_in_repo('answer.csv', 'PGEHackathon/hidden')
    solution_df = pd.read_csv(StringIO(result))
    solution_array = solution_df.sort_values(by=['Masked Well Name', 'Fuel Type']).iloc[:, 2].to_numpy()


    team_names = []
    team_mape = []
    team_goodness_score = []
    for repo in repos:


        if repo not in blocked_list:
            print(f"Generating Report For: {repo}")

            result = gh.get_file_in_repo('solution.csv', repo)

            if result is not None:

                prediction_df = pd.read_csv(StringIO(result))
                prediction_df = prediction_df.sort_values(by=['Masked Well Name', 'Fuel Type'])

                if prediction_df["Fuel Value"].mean() == 1:
                    continue 

                team_name = repo.split('/')[1]
                team_names.append(team_name)

                mape = create_accuracy_plot_and_return_mape(prediction_df, 
                                                            solution_array)
                team_mape.append(mape)

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


                create_team_report(team_name, mape, 
                                   goodness_score, 
                                   presentation_comments,
                                   code_review_comments)

    df = pd.DataFrame(np.array([team_names, 
                                team_mape, 
                                team_goodness_score]).T, columns=['Team Names',
                                                                  'MAPE',
                                                                  'Goodness Score'])
    #df['Team Names'].to_csv('team_names.csv', index=False)

    df['Short Names'] = df['Team Names'].apply(parse_team_name)
    df['Team Names'] = df['Team Names'].apply(parse_team_name)
    df.set_index(['Short Names'], inplace=True)

    df['Pres. Score'] = presentation_score_df
    df['Code Score'] = code_review_score_df
    df['MAPE Rank'] = df['MAPE'].astype('float64').rank(method='min', ascending=True, na_option='top')
    df['Goodness Rank'] = df['Goodness Score'].astype('float64').rank(method='min', ascending=False, na_option='top')
    df['Pres. Rank'] = df['Pres. Score'].astype('float64').rank(method='min', ascending=False, na_option='top')
    df['Code Rank'] = df['Code Score'].astype('float64').rank(method='min', ascending=False, na_option='top')

    df['Overall Rank'] = (0.375 * df['MAPE Rank'].astype('float64') + 
                          0.375 * df['Goodness Rank'].astype('float64') + 
                          0.200 * df['Pres. Rank'].astype('float64') + 
                          0.050 * df['Code Rank'].astype('float64')
                         ).rank(method='min')

    df.sort_values('Overall Rank', inplace=True)
    df.index.name = None
    #df.reset_index(inplace=True)
    with open('final_rankings_table.tex', 'w') as f:
        df.to_latex(index=False, buf=f)

    subprocess.run(['latexmk', '-pdflua', 
                    '-output-directory=reports', 'final_report.tex'])
