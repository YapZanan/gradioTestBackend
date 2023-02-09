FROM python:3.9.0

WORKDIR /user/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=app.py

EXPOSE 5000

ENTRYPOINT [ "python", "app.py" ]

CMD ["flask", "run", "--host=0.0.0.0"]
