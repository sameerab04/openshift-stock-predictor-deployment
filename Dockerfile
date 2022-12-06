FROM python:3.8-bullseye

# we probably need build tools?
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
    build-essential

# copy the requirements file into the image
COPY ./* /app/


# switch working directory
WORKDIR /app

# first: install all required packages for pystan
RUN pip install numpy cython pystan==2.19.1.1

# third: install prophet itself
RUN pip install --no-cache-dir --upgrade prophet

RUN pip install yfinance statistics

RUN pip install -U Flask


ENTRYPOINT ["python", "./prophet_predication.py"]
