# 1. Use the official lightweight Python base image
FROM python:3.11-slim

# 2. Set WOrking directory inside the container

WORKDIR /app

# 3. Copy only dependency file first (for Docker caching)
# the dot means we bring everything of the requirement file 

COPY requirements.txt .    

# 4. Install Python dependencies 
RUN pip install --upgrade pip \
&& pip install -r requirements.txt \
&& apt-get clean && rm -rf /var/lib/apt/lists/*

# 5. Copy the entire project into the image
COPY . .

# 6. Expose FastAPI port
expose 8000

#7. Run the FastAPI app using uvicorn (change path if needed)
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

