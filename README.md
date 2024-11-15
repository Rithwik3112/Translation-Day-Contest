Here's a step-by-step guide for a new user to start with the provided code and set up the Translation Quality Checker application:

**Step 1: Install Prerequisites**
Ensure that the following software and tools are installed on your system:

-- Python 3.7 or later (preferred version: 3.9+)
-- pip (Python package manager)

**Step 2: Clone or Download the Code**
Clone from a Git Repository (if applicable):
  git clone <repository-link>
  cd <repository-folder>
              or
Download as ZIP:
If the code is shared as a ZIP file, download and extract it to your desired directory.

**Step 3: Set Up a Virtual Environment**
A virtual environment helps keep the dependencies isolated.

Create a virtual environment:
  python -m venv venv

Activate the virtual environment:
On Windows:
  venv\Scripts\activate

On macOS/Linux:
  source venv/bin/activate

**Step 4: Install Dependencies**

Install the required Python packages by running:
  pip install -r requirements.txt

**Step 5: Configure the Azure Translator API**
Obtain an API key and endpoint URL for Azure Translator from the Azure Portal.
Update the placeholder variables in the model.py file.

**Step 6: Run the Flask Application**
Start the Flask development server:
  python app.py
  
The terminal should display something like:
  Running on http://127.0.0.1:5000/
  
Open your web browser and navigate to http://127.0.0.1:5000/.

**Step 7: Test the Application**
Enter Source Text and Human Translation in the input fields provided on the webpage.
Click the Submit button.
The backend will process the input and display the evaluation results.
