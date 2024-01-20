#!/usr/bin/env python#!/usr/bin/env python

import requests
import math
import base64


class Github(object):
    def __init__(self, token):

        self.headers = {
            "Authorization": "token " + token 
        }

    def get_number_of_repos(self, organization, repo_type="all"):

        response = requests.get(
            "https://api.github.com/orgs/{}".format(organization, repo_type),
            headers=self.headers,
        )

        private_repos = response.json()["total_private_repos"]
        public_repos = response.json()["public_repos"]

        if repo_type == "public":
            return public_repos
        elif repo_type == "private":
            return private_repos
        elif repo_type == "all":
            return public_repos + private_repos

    def get_organization_repos(self, organization, repo_type="all", exclude_archived=True):

        number_of_repos = self.get_number_of_repos(organization, repo_type=repo_type)

        repos = []
        for i in range(1, math.floor(number_of_repos / 100) + 2):

            response = requests.get(
                "https://api.github.com/orgs/{}/repos?type={}&page={}&per_page=100".format(
                    organization, repo_type, i
                ),
                headers=self.headers,
            )

            for item in response.json():
                if exclude_archived:
                    if not item["archived"]:
                        repos.append(item["full_name"])
                else:
                    repos.append(item["full_name"])

        return repos

    def delete_organization_repos(self, organization, repo_type="private"):

        repos = self.get_organization_repos(organization, repo_type=repo_type)

        for repo in repos:

            response = requests.delete(
                "https://api.github.com/repos/{}".format(repo), headers=self.headers
            )

        return
    
    def delete_repos(self, repos):

        for repo in repos:

            response = requests.delete(
                "https://api.github.com/repos/{}".format(repo), headers=self.headers
            )

        return
    
    def get_file_sha(self, filename, repo):
    
        result = requests.get("https://api.github.com/repos/{}/contents/{}".format(repo, filename),       
                              headers=self.headers)
    
        return result.json()['sha']
    
    
    def update_file_in_repo(self, filename, repo):
        
        sha = self.get_file_sha(filename, repo)
        
        with open(filename, 'rb') as f:
            byte_content = f.read()
            base64_bytes = base64.b64encode(byte_content)
            base64_string = base64_bytes.decode('utf-8')
            
            payload = {"message": "Replace/Update {}".format(filename),
                       "author": {"name": "John T. Foster", "email": "john.foster@utexas.edu"},
                       "sha": sha,
                       "content": base64_string}
            
            result = requests.put("https://api.github.com/repos/{}/contents/{}".format(repo, filename), 
                          headers=self.headers, 
                          json=payload)
            
        return 

    def get_file_in_repo(self, filename, repo):
        
        result = requests.get("https://api.github.com/repos/{}/contents/{}".format(repo, filename), 
                      headers=self.headers)

        if result.status_code == 200:
            byte_content = result.json()['content']
            base64_bytes = base64.b64decode(byte_content)
            base64_string = base64_bytes.decode('utf-8')
            return base64_string
        else:
            return None

            
    
        # with open(filename, 'rb') as f:
            # byte_content = f.read()
            # base64_bytes = base64.b64encode(byte_content)
            # base64_string = base64_bytes.decode('utf-8')
            
            # payload = {"message": "Replace/Update {}".format(filename),
                       # "author": {"name": "John T. Foster", "email": "john.foster@utexas.edu"},
                       # "sha": sha,
                       # "content": base64_string}
            
        return result
    
    
    def add_file_to_repo(self, filename, repo):
        
        with open(filename, 'rb') as f:
            byte_content = f.read()
            base64_bytes = base64.b64encode(byte_content)
            base64_string = base64_bytes.decode('utf-8')
            
            payload = {"message": "Add {}".format(filename),
                       "author": {"name": "John T. Foster", "email": "john.foster@utexas.edu"},
                       "content": base64_string}
            
            result = requests.put("https://api.github.com/repos/{}/contents/{}".format(repo, filename), 
                          headers=self.headers, 
                          json=payload)
            
        return
    
    
    def delete_file_in_repo(self, filename, repo):
        
        sha = self.get_file_sha(filename, repo)

            
        payload = {"message": "Delete {}".format(filename),
                   "author": {"name": "John T. Foster", "email": "john.foster@utexas.edu"},
                   "sha": sha}
            
        result = requests.delete("https://api.github.com/repos/{}/contents/{}".format(repo, filename), 
                          headers=self.headers, 
                          json=payload)
            
        return
    
    def filter_repos(self, organization, pattern, repo_type='all'):
        
        all_repos = self.get_organization_repos(organization, repo_type)
        
        filtered_repos = []
        
        for repo in all_repos:
        
            if pattern in repo:
                filtered_repos.append(repo)
            
        return filtered_repos

    def get_workflow_runs(self, repo):

        result = requests.get("https://api.github.com/repos/{}/actions/runs".format(repo), 
                          headers=self.headers)

        return result.json()

    def get_latest_workflow_run(self, repo):
        
        runs = self.get_workflow_runs(repo)

        if runs['total_count'] == 0:
            return None
        else:
            return runs['workflow_runs'][0]

    def get_latest_workflow_conclusion(self, repo):
        
        run = self.get_latest_workflow_run(repo)

        if run is None:
            return None
        else:
            return run['conclusion']

    def get_latest_workflow_commit_time_and_conclusion(self, repo):
        
        run = self.get_latest_workflow_run(repo)

        if run is None:
            return (None, None)
        else:
            return (run['head_commit']['timestamp'], run['conclusion'])

    def change_repo_name(self, old, new):

        requests.patch("https://api.github.com/repos/{}".format(old), 
                       data={'name': str(new)}, headers=self.headers)

        return

    def get_pull_requests(self, repo):

        result = requests.get("https://api.github.com/repos/{}/pulls".format(repo), 
                          headers=self.headers)

        return result.json()

    def get_pull_request_ids(self, repo):

        prs = self.get_pull_requests(repo)

        ids = []
        for pr in prs:
            ids.append(pr['id'])

        return ids

    def merge_all_pull_requests(self, repo):

        prs = self.get_pull_requests(repo)

        for pr in prs:
            if pr['state'] == 'open':
                pull_number = pr['number']

                payload = { "commit_title": "Merge PR {}".format(pull_number),
                            "merge_method": 'squash'}

                result = requests.put("https://api.github.com/repos/{}/pulls/{}/merge".format(repo, pull_number), 
                                      headers=self.headers, json=payload)
