# Intelligent Procurement Assistant for Real-Time Data Insights

![AI in Procurement](./images/Intelligent_Procurement_Assistant.png)

### Problem Description
Organizations managing extensive procurement operations often face challenges in efficiently extracting insights from large amounts of purchasing data. Important metrics such as order volumes, high-spending periods, and frequently ordered items are critical for informed decision-making, but manually analyzing this data can be labor-intensive and time-consuming.

This project addresses these challenges by developing an intelligent chat assistant that automates the retrieval of procurement insights. The assistant allows users to ask procurement-related questions and receive immediate, data-driven responses, empowering procurement teams to make faster and more informed decisions.

### Objective
To create a prototype assistant capable of answering queries related to procurement data. This includes:
- Total number of orders within specified periods.
- Identifying the quarter with the highest spending.
- Analyzing frequently ordered line items.
- Answering other relevant procurement-related inquiries.


## Project Overview
The Intelligent Procurement Assistant is a virtual assistant designed to streamline data-driven decision-making within procurement processes. This assistant allows procurement teams to interact with procurement data directly, answering questions about order trends, spending patterns, and frequently ordered items. By automating these insights, the assistant saves time and reduces the need for manual data analysis, empowering users to make faster and more informed purchasing decisions. 

## Dataset 
The dataset used in this project contains detailed procurement data, providing insights into purchasing trends, order volumes, and spending patterns. Key data points include:

- **Order Details**: Information on individual orders, including order numbers, dates, and amounts spent.
- **Time Periods**: Dates and timestamps that enable analysis by month, quarter, or year.
- **Items and Categories**: Details about the items purchased, including item descriptions, categories, and frequency of orders.
- **Spending Information**: Data on total spending per order, which allows for tracking of high-spending periods and analysis of spending trends over time.

This structured dataset is essential for powering the Intelligent Procurement Assistant you can find it through this link: https://www.kaggle.com/datasets/sohier/large-purchases-by-the-state-of-ca

## Technologies

- **Python 3.12**: The core programming language used for implementing the assistant's functionality, data processing, and handling user queries.
- **Docker and Docker Compose**: Used for containerization, ensuring that the project runs consistently across different environments by packaging dependencies and configurations together.
- **pgvector**: A PostgreSQL extension employed for efficient full-text search and similarity search, enabling quick retrieval of relevant information from the procurement data.
- **Flask**: A lightweight web framework used to create a user-friendly API that facilitates interaction between users and the assistant.
- **multilingual-e5-small**: A model used to generate high-quality embeddings, enabling semantic search across the dataset for more accurate query responses.
- **Qwen/Qwen2.5-1.5B (via Hugging Face)**: A large language model used for generating answers to user queries, adding conversational intelligence to the assistant.

