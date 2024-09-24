FROM --platform=linux/amd64 python:3.12.6-slim-bookworm
# RUN curl https://bootstrap.pypa.io/get-pip.py | python
# RUN python -m pip install --upgrade pip
RUN pip install pandas==2.2.3 && \
    pip install numpy==2.1.1 && \
    pip install scikit-learn==1.5.2
COPY . /app
WORKDIR /app
# EXPOSE 80
CMD ["python3", "--version"]