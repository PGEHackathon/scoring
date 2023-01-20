#!/usr/bin/env python

import scipy.stats
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jinja2
from jinja2 import Template
import os
import subprocess
import matplotlib
import json

pgf_with_latex = {"pgf.texsystem": 'pdflatex'}
matplotlib.rcParams.update(pgf_with_latex)

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

def create_team_report(team_name, score, 
                       presentation_comments,
                       code_review_comments):

    template = latex_jinja_env.get_template('report_template.tex')

    #Render tex and write to file
    rendered_tex = template.render(teamname=team_name.replace('_', ' '), 
                                   score=score,
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
                    'PGEHackathon/resources', 'PGEHackathon/hidden',
                    'PGEHackathon/truth_data', 'PGEHackathon/submissions',
                    'PGEHackathon/johntfoster', 'PGEHackathon/simulation_results']

    team_names = []
    team_score = []

    print(f"Getting answers file.")
    result = gh.get_file_in_repo('answer.csv', 'PGEHackathon/hidden')
    answer_df = pd.read_csv(StringIO(result))

    for repo in repos:

        if repo not in blocked_list:

            print(f"Collecting submission file for: {repo}")
            try: 
                result = gh.get_file_in_repo('solution.csv', repo)
            except:
                print(f"Failed for {repo}")
                result = None

            team_name = repo.split('/')[1]

            if result is not None:

                submission_df = pd.read_csv(StringIO(result))

                team_names.append(team_name)

                result_df = pd.read_csv(StringIO(result))

                accuracy = accuracy_score(answer_df, result_df)
                score = 2 * (accuracy - 1/2.)
                team_score.append(score)

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


                create_team_report(team_name, score, 
                                   presentation_comments,
                                   code_review_comments)

    df = pd.DataFrame(np.array([team_names, 
                                team_score, 
                                team_total_prod]).T, columns=['Team Names',
                                                              'Score'])
    df['Short Names'] = df['Team Names'].apply(parse_team_name)
    df.set_index(['Short Names'], inplace=True)

    df['Pres. Score'] = presentation_score_df
    df['Code Score'] = code_review_score_df
    df['Score Rank'] = df['Score'].astype('float64').rank(method='min', ascending=True, na_option='top')
    df['Total Production Rank'] = df['Field total production'].astype('float64').rank(method='min', ascending=False, na_option='top')
    df['Pres. Rank'] = df['Pres. Score'].astype('float64').rank(method='min', ascending=False, na_option='top')
    df['Code Rank'] = df['Code Score'].astype('float64').rank(method='min', ascending=False, na_option='top')

    df['Overall Rank'] = (0.75 * df['Score Rank'].astype('float64') + 
                          0.2 * df['Pres. Rank'].astype('float64') + 
                          0.05 * df['Code Rank'].astype('float64')
                         ).rank(method='min')

    df.sort_values('Overall Rank', inplace=True)
    df.index.name = None
    #df.reset_index(inplace=True)
    with open('final_rankings_table.tex', 'w') as f:
        df.to_latex(index=False, buf=f)

    subprocess.run(['latexmk', '-pdf', 
                    '-output-directory=reports', 'final_report.tex'])
