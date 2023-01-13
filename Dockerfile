FROM python:3.9

COPY . /rohit/app/

EXPOSE 3000

WORKDIR /rohit/app/

RUN pip install -r requirements.txt

CMD python app.py