<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



# Cortical Hippocampal model

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
     <a href="#todos">ToDOs</a>
    </li>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## TODOs
1. Audio data testing
  - test audio data training on clustr with all data
  - clean and check code
  - check model?
  - look at some results from the trained model

2. Then integrate a second input stream from videos
2. Integrate Mashbayars model instead of simple GPU, modify input?


<!-- ABOUT THE PROJECT -->
## About The Project




<!-- GETTING STARTED -->
## Getting Started


### Prerequisites

How to use the project on the cluster. Load the following modules:

```sh
$ module load scipy-stack/2024b python/3.11.5 opencv/4.10.0
```
Create and activate a new virtual env. And update pip
```sh
$ virtualenv --no-download corhip
$ source corhip/bin/activate
$ pip install --no-index --upgrade pip
```

Further, please install: 
```sh
$ pip install --no-index torch torchvision torchaudio
```

### Installation

1. Load data
2. Load project
   ```sh
   git clone ...
   ```


<!-- USAGE EXAMPLES -->
## Usage

Use cfg files to execute specific script.
$ python main.py



<!-- ROADMAP -->
## Roadmap



<!-- CONTACT -->
## Contact

Clara Kümpel - [@clarakumpel](https://twitter.com/clarakumpel) - clara.kumpel@mila.quebec

<!-- Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name) -->



