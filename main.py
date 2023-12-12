import logging
import ast
import os
import streamlit as st

from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = st.secrets["OPENAI_KEY"]
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI API KEY")

# Function to process the PDF and return a sample JSON blob
def process_pdf(file):
    # Upload PDF, grab text
    reader = PdfReader(file)
    allpagetext = []
    for i in reader.pages:
        allpagetext.append(i.extract_text())

    # Page by page, search for "Prior Authorization" , identify all page numbers +/- 1
    relpages = []
    for page in range(len(allpagetext)):
        #pagetext = allpagetext[page]
        #pagetext = unicodedata.normalize('NFKC', pagetext)
        #pagetext = pagetext.replace(' ', '')
        #if "PriorAuthorization" in pagetext or "Priorauthorization" in pagetext or "priorauthorization" in pagetext:
        #  relpages.append([max(0, page-1), page, min(page+1, len(allpagetext))])
        if "Prior Authorization" in allpagetext[page]:
            relpages.append([page-1, page, page+1])
    chat = ChatOpenAI(temperature=0,openai_api_key=OPENAI_API_KEY, model_name="gpt-4", max_tokens=1500)
    instruct = """
    You are a healthcare insurance policy bot.

    Your goal is to analyze insurance policy documents and identify all services that require Prior Authorization.

    You must focus solely on identifying specific services (with full descriptions) that require (or may require) Prior Authorization.
    """

    prompt = """
    Below you will see an excerpt from a health insurance policy document. The excerpt might NOT contain any services requiring Prior Authorization.
    It has been identified because it has the key words "Prior Authorization", regardless of context. \

    Your goal is to identify whether the specific instance contains information regarding Prior Authorization of a specific
    service or not.

    If the excerpt DOES specify Prior Authorization being required for specific service (keep in mind the phrasing might be "is required" or "maybe required" or "may be required for certain services"), your expected output is:
    {'Service': name_of_service_here , 'Details': description_of_service_here}

    Example with Prior Authorization required:
    Text: "Ambulance services
    Covered ambulance services, whether for an emergency or non-emergency situation, include fixed wing, rotary wing, and ground
    ambulance services, to the nearest appropriate facility that can
    provide care only if they are furnished to a member whose
    medical condition is such that other means of transportation could
    endanger the person’s health or if authorized by the plan. If the
    covered ambulance services are not for an emergency situation, it
    should be documented that the member’s condition is such that
    other means of transportation could endanger the person’s health
    and that transportation by ambulance is medically required.
    $200 copay per one-way
    trip (includes ground and
    air transport) for each
    Medicare-covered
    ambulance service.
    *Prior Authorization
    required for non-emergency needs."
    Your Expected Response: "{'Service': "Ambulance Services",
    'Details': "Covered ambulance services, whether for an emergency or non-emergency situation, include fixed wing, rotary wing, and ground
    ambulance services, to the nearest appropriate facility that can
    provide care only if they are furnished to a member whose
    medical condition is such that other means of transportation could
    endanger the person’s health or if authorized by the plan. If the
    covered ambulance services are not for an emergency situation, it
    should be documented that the member’s condition is such that
    other means of transportation could endanger the person’s health
    and that transportation by ambulance is medically required.
    $200 copay per one-way
    trip (includes ground and
    air transport) for each
    Medicare-covered
    ambulance service. *Prior Authorization
    required for non-emergency needs."}

    In the example above, notice how there is a specific service named, described and then followed by standard text that says this service DOES require Prior Authorization (sometimes with stipulations like "for non-emergency needs").
    As given in the example above, you MUST include the specific Prior Authorization stipulations in your response (such as "Prior Authorization is required", "Prior Authorization may be required", "Prior Authorization required for non-emergency needs" etc)

    If the excerpt mentions Prior Authorization not in relation to a specific service but in general discussion, your expected output is:
    NA

    Example with Prior Authorization mentioned without specific services:
    Text: "We will arrange for any medically necessary covered benefit outside of our provider
    network, but at in-network cost sharing, when an in-network provider or benefit is
    unavailable or inadequate to meet your medical needs. Your PCP or specialist is
    responsible for obtaining Prior Authorization, but you should confirm with your PCP or
    specialist that authorization was requested"
    Your Response: "NA"

    Notice in the above example that even though "Prior Authorization" is mentioned, but it's not in reference to any specific service. It's just a general discussion.

    Actual Text:
    """
    
    #gptresponses = []
    #for option in relpages[1:4]:
    #    alltext = allpagetext[option[0]] + allpagetext[option[1]] + allpagetext[option[2]]
    #    messages = [
    #        SystemMessage(content=instruct),
    #        HumanMessage(content=prompt + alltext)
    #    ]
    #    response = chat(messages)
    #    gptresponses.append(response.content)
    

    # Function to perform the task for each iteration
    def process_option(option, instruct, prompt, allpagetext):
        alltext = allpagetext[option[0]] + allpagetext[option[1]] + allpagetext[option[2]]
        messages = [
            SystemMessage(content=instruct),
            HumanMessage(content=prompt + alltext)
        ]
        response = chat(messages)
        return response.content
    
    gptresponses = []
    num_workers = 8  # Number of workers
    tasks = relpages  # Assuming there are 5 tasks
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Create a future for each task
        futures = [executor.submit(process_option, task, instruct, prompt, allpagetext) for task in tasks]

        # Wait for each future to complete and collect results
        for future in as_completed(futures):
            gptresponses.append(future.result())
    
    finalres = []
    for i in gptresponses:
        if "NA" not in i:
            if "\n\n" in i:
                splitresponses = i.split("\n\n")
                for j in splitresponses:
                    finalres.append(ast.literal_eval(str(j)))
            else:
                finalres.append(ast.literal_eval(str(i)))

    # For demonstration, returning a static JSON blob
    # In a real-world application, this should be generated based on the PDF content
    #json_blob = {
    #    "services": [
    #        {"Service": "Service 1", "Details": "Details of Service 1"},
    #        {"Service": "Service 2", "Details": "Details of Service 2"},
    #        # Add more services as needed
    #    ]
    #}
    json_blob = {
        "services": finalres
    }
    return json_blob

# Streamlit app main function
def main():
    st.title('EquallHealth Plan Extractor - Prior Authorization')

    # File uploader for a single PDF file
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Check if a file is uploaded
    if uploaded_file:
        st.write(f"Results for {uploaded_file.name}:")
        try:
            # Process the PDF
            result = process_pdf(uploaded_file)
            
            # Display dropdowns based on the JSON blob
            for service in result["services"]:
                with st.expander(service["Service"]):
                    st.write(service["Details"])
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")


if __name__ == "__main__":
    main()
