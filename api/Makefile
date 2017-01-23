NAME=docker-scikit-api

build:
	docker build -t $(NAME) .

run:
	docker run -it --rm=true $(NAME) bash

remove:
	-docker stop $(NAME)
	-docker rm $(NAME)

run_container:
	docker run -p 8080:8080 --rm -v /Users/nicktgr15/workspace/docker-scikit/charsearch/vectorizer.pickle:/tmp/vectorizer.pickle  -v /Users/nicktgr15/workspace/docker-scikit/charsearch/model.pickle:/tmp/model.pickle docker-scikit-api python api.py
