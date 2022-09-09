FROM python:3.8

RUN mkdir mr_acr
COPY . mr_acr 

ENV QT_API pyqt5
ENV MPLBACKEND Agg

RUN pip install pipenv 

ENV PIPENV_PIPFILE=/mr_acr/Pipfile
RUN pipenv sync 
