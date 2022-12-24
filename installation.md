---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1

kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---


# Installation 

We will use jupyter-book as a verbose form of the lecture slides. This is essentially a collective lecture notes, scribed by the students. The source code for the the book is hosted in the [github repo](https://github.com/swing-research/prbook). 

## Local build
To  run and build the course jupyter-book locally, you need to first install [jupyter-book](https://jupyterbook.org/en/stable/start/overview.html). You can install juptyerbook using the pip command:
```
pip install -U jupyter-book
```

If you are using conda, you can install it using:

```
conda install -c conda-forge jupyter-book
```
For more details regarding the installation  you can check the [link](https://jupyterbook.org/en/stable/start/overview.html). 

We assume you have python and some of the machine learning libraries used in the course installed. Next, you need to download the repository in your local machine. Run the below command to clone the repository in your machine 

```
git clone git@github.com:swing-research/prbook.git
```
The above command downloads the course jupyter-book files from Github into your local directory. This might take some times as there are several media files to be downloaded. Once downloaded you will have a folder called prbook. The prbook folder should look something like: 

```
./prbook
├── book_data
├── _build
├── _config.yml
├── denoise-deblur.md
├── filtering.md
├── images
├── installation.md
├── introduction.md
├── intro.md
├── linear-methods.md
├── logo.png
├── logo_sada-lab_black.svg
├── modeling-knowledge.md
├── perceptron.py
├── __pycache__
├── references.bib
├── requirements.txt
└── _toc.yml

```

You are ready to build prbook and generate the HTML files. Now run the following command: 

```
jupyter-book build prbook
```
This will generate a functional HTML site. The site is placed in  ```./prbook/_build/```. You can open ```index.html``` in the directory ```/prbook/_build/html/``` folder to view the contents of the book. For more details on building and troubleshooting jupyter-book builds follow the [link](https://jupyterbook.org/en/stable/start/overview.html) 


## Scribing 
This is an optional task for students to get extra credit or for people who missed some lectures with recorded attendance. You will be assigned certain lectures to scribe. Based on the assigned lectures you need to fill the corresponding Markdown document with python code snippets and explanations pertaining to the lecture. If a file  for the lecture is not present you are free to create a new file with an appropriate name, however it is better to give us a prior notification. The file's name must then be added in the proper sequence to  ```_toc.yml```. You can use [Introduction](./introduction.md), [Denoising](./denoise-deblur.html#denoising), and [Wiener filter](./denoise-deblur.html#wiener-filter) as examples of the tone and level of each topic's explanations.

### Editing and Style Convention
The prefered way of editing and creating the lecture notes is through Markdown files as done in ```prbook```. But you can create the books using jupyter notebook as well.  You can look at ```./prbook/introduction.md``` on how to write down the markdown files which contains python snippets from the slides and explanation. Latex macros from ```./prbook/_config.yml```, should be used for equations in the explanation part of the text. Use Latex for equations and math terms. Follow the notation scheme in  [Denoising and deblurring](./denoise-deblur.md) whenever possible. 

### Submitting 
The best method for uploading your work is to submit it as a pull request (PR), complete with your modifications. A sample PR is provided [below](#pull-request-Example ) Your commit messages should be detailed. After reviewing your changes, we will decide whether to accept them as is or ask you to make some improvements. You will receive full credit for the assignment given to you if the PR is examined and merged into the main branch, marking the task as successful.

Email is another option for submitting your work. We assess it after that and let you know if it is appropriate or not.


### Pull request Example

Here, we'll take a look at an example pull request for editing [Installation](./installation.md) to change a phrase. You should first launch your browser and access the [github repo](https://github.com/swing-research/prbook). For you to contribute your changes, a github account is required. You must fork the repository once you are on the prbook gihtub page (click the ```Fork``` button in the top right corner). The repository is duplicated in your account as a result. You must clone your copy of the reposity onto your local machine after it has been forked. The syntax for the clone command is:
```
git clone git@github.com:< your github username >/prbook.git
```
You will have the personal copy of ```prbook``` downloaded in your local machine.  Open the file ```./prbook/installation.md``` and go to line '86' and change the text from '\<your github username\>' to '\<username\>' or you can change it your github user name as well. Once changed, you need push the local changes to the github server. You can run the following command to push the changes. 


```
git add ./instruction.md 
git commit -m 'changing the user name in line 89' 
git push
```

**NOTE:**
Be specific in your commit messages, this helps in debugging if there are any errors.

The above command uploads the local changes to your copy of prbook in the github server. You must now access your copy of the prbook on your github page. Your copy's hyperlink should read ```https://github.com/<username>/prbook```.  The modifications are also visible in the browser. Now you are ready to push the changes to the main jupyter-book. There is a button labeled ```contribute``` on the browser. This button provides you the choice to start a pull request when you click it. You are taken to the 'Open a pull request' page when you click this.  You get an option to add title and to comment on your changes. Once done, click on the ```Create pull request``` button.  Now your pull request is submmitted.  You can check the comments regarding your updates in the [pull request tab](https://github.com/swing-research/prbook/pulls) after a reviewer has been assigned to it. 













