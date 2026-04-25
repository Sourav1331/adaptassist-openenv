FROM python:3.10

WORKDIR /app

# copy everything
COPY . .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt || cat requirements.txt
# expose port (IMPORTANT for HF)
EXPOSE 7860

# run app
CMD ["python", "app.py"]