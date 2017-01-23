FROM ubuntu:14.04

RUN apt-get update
RUN apt-get install wget build-essential git python -y

RUN apt-get install python-pip python-dev -y
RUN apt-get install libicu-dev -y

RUN pip install --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
ADD src /opt/scikit-api

RUN python -c 'from polyglot.downloader import downloader; downloader.download("TASK:transliteration2")'

RUN chmod 777 -R /opt/scikit-api
WORKDIR /opt/scikit-api
EXPOSE 55000-55099

#RUN ls -l /tmp
#COPY src /opt/scikit-api
#WORKDIR /opt/scikit-api
