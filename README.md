# Intelligent Conversational Chatbot

## Project Description
This project is an intelligent conversational chatbot implemented using BERT (Bidirectional Encoder Representations from Transformers). The chatbot is designed to understand and respond to various user intents in a conversational manner. The different intents include greetings, questions about the time, identity of the user, chatbot, jokes, queries, and more.

## Training the Model
### Preparing the Training Data
The training data consists of various user queries and their corresponding intents. The data is preprocessed and tokenized to be compatible with the BERT model.

### Training Process
1. Open the Jupyter notebook `Intent_Classification_Chatbot.ipynb`.
2. Follow the steps in the notebook to preprocess the data, train the model, and save the trained model as a `.pth` file.
3. The trained model will be saved as `data.pth`.

## Running the Gradio App
Gradio is used to create a user-friendly web interface for interacting with the chatbot.

### Steps to Run the Gradio App
1. Ensure you have all the required dependencies installed:
    ```bash
    pip install -r requirements.txt
    ```
2. Run the Gradio app:
    ```bash
    python run.py
    ```
3. Open the provided URL in your web browser to interact with the chatbot.

## Further Extensions
- **Additional Intents**: Add more intents to make the chatbot more versatile.
- **Improved Responses**: Enhance the response generation to make the chatbot more engaging.
- **Deployment**: Deploy the chatbot on a cloud platform for wider accessibility.

## Contributing Guidelines
1. Fork the repository.
2. Create a new branch for your feature or bugfix:
    ```bash
    git checkout -b feature-name
    ```
3. Commit your changes:
    ```bash
    git commit -m "Description of your changes"
    ```
4. Push to the branch:
    ```bash
    git push origin feature-name
    ```
5. Create a pull request.

## Contact Information
For any questions or suggestions, please contact:
- Name: Mehul Mathur
- Email: mathurmehul3@gmail.com
