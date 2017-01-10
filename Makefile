NAME=docker-scikit
NAME_DOCKER_HUB=nicktgr15/docker-scikit

build:
	docker build -t $(NAME) .

build_docker_hub:
	docker build -t $(NAME_DOCKER_HUB) .

push_docker_hub:
	docker push nicktgr15/docker-scikit

run:
	docker run -it --rm=true $(NAME) bash

remove:
	-docker stop $(NAME)
	-docker rm $(NAME)

env:
	virtualenv env
	. env/bin/activate && pip install -r requirements.txt

run_container:
	docker run --rm -v /Users/nicktgr15/workspace/docker-scikit/20news-bydate:/Users/nicktgr15/workspace/docker-scikit/20news-bydate -v /Users/nicktgr15/workspace/docker-scikit:/Users/nicktgr15/workspace/docker-scikit  docker-scikit python /opt/docker-scikit/run.py  --input-dir /Users/nicktgr15/workspace/docker-scikit/20news-bydate --output-dir /Users/nicktgr15/workspace/docker-scikit

download_dataset:
	wget http://people.csail.mit.edu/jrennie/20Newsgroups/20news-bydate.tar.gz