FROM python:3.6

RUN mkdir /var/www

WORKDIR /var/www

COPY requirements.txt ./

RUN pip install -r requirements.txt

WORKDIR /var/www/src

CMD ["uwsgi", "--ini", "/var/www/uwsgi.ini"]