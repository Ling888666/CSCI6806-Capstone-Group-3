FROM --platform=linux/amd64 python:3.13.0-slim-bookworm
# RUN curl https://bootstrap.pypa.io/get-pip.py | python
# RUN python -m pip install --upgrade pip
RUN pip install python-dateutil==2.9.0.post0 && \
    pip install numpy==2.1.2 && \
    pip install pandas==2.2.3 && \
    pip install xgboost==2.1.2 && \
    pip install scikit-learn==1.5.2
COPY . /app
WORKDIR /app
# EXPOSE 80
CMD ["python3", "--version"]