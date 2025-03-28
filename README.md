# **Cyberbullying Detection Using Twitter Hate Speech Data**  

This project aims to detect cyberbullying in text data using a **deep learning model** trained on Twitter hate speech data. The data is stored in **MongoDB Atlas**, and the model integrates **Bidirectional GRU, Attention Mechanism, and a Capsule Network (CapsNet)** to classify text as **"Cyberbullying" or "Not Cyberbullying"**.  


## **Project Overview**  

- Cyberbullying has become a significant concern on social media platforms, particularly Twitter, where anonymity and ease of access contribute to harmful online interactions. This project focuses on **automated cyberbullying detection** using a **deep learning model** trained on hate speech data from Twitter.  
- The project employs **natural language processing (NLP) techniques** and **neural networks** to analyze text data and classify it as either **"Cyberbullying" or "Not Cyberbullying"**.  
- The dataset, containing labeled comments, is stored in **MongoDB Atlas**, ensuring scalability and easy access for real-time analysis. Users can either input their own text or use pre-existing labeled Twitter hate speech data to train the model. The trained model is then used to classify new comments, providing a binary output (0 = Not Cyberbullying, 1 = Cyberbullying).  
- By leveraging deep learning and NLP, this project aims to contribute to the **fight against online harassment** by enabling automated moderation and flagging of harmful content.  


## **Model Architecture**  

The cyberbullying detection model incorporates a **hybrid deep learning approach** to effectively capture contextual meaning and relationships in text. The architecture consists of the following key components:  

1. **Bidirectional GRU (Gated Recurrent Unit)**  
   - Processes the text sequence in **both forward and backward** directions to capture contextual dependencies.  
   - Helps retain long-range dependencies within the input text.  

2. **Attention Mechanism**  
   - Focuses on the most **important words or phrases** in a sentence.  
   - Enhances interpretability by identifying key sections contributing to classification.  

3. **Capsule Network (CapsNet)**  
   - Preserves **spatial relationships** between words.  
   - Uses **dynamic routing** to capture hierarchical patterns in the text.  

### **Model Output:**  
- The final layer outputs a **binary prediction**:  
  - `0 = Not Cyberbullying`  
  - `1 = Cyberbullying`  
- The trained model is saved as `cyberbullying_model.h5` for future inference.  


## **Project Setup & Execution**  

Follow these steps to set up and run the project:  

### **1. Clone the Repository**  

git clone https://github.com/NugooruTaruni/Cyberbullying-Detection.git && cd Cyberbullying-Detection

### **2. Create a Virtual Environment**
Windows: python -m venv venv && venv\Scripts\activate
Mac/Linux: python3 -m venv venv && source venv/bin/activ

### **3. Install Required Dependencies**
pip install -r requirements.txt

### **4. Set Up MongoDB Atlas**
- Create a MongoDB Atlas account and a database CBDetection with a collection Comments.
- Update the MongoDB connection string in db.py.

### **5. Run the Application**
python app.py
