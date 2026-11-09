# AI-Powered Personalized Travel Planner Using NLP

An AI-powered travel planning application that generates personalized,
day-wise itineraries based on a user's destination, interests, trip
duration, and budget.

The project combines Python, Streamlit, a structured CSV dataset, and
the locally hosted Llama 3.2 model through Ollama.

## Features

-   **Personalized itinerary generation:** Generates a day-wise travel
    plan using the user's destination, interests, duration, and budget.
-   **AI Travel Assistant:** Accepts natural-language travel questions.
-   **Dataset-based recommendations:** Uses attraction information
    stored in a CSV dataset.
-   **Recommendation validation:** Checks generated attraction names
    against available dataset records and helps prevent invalid or
    duplicate recommendations.
-   **Budget estimation:** Calculates estimated entry fees, transport,
    food, and accommodation costs.
-   **Approximate route information:** Uses attraction coordinates and
    the Haversine formula to estimate distances. These are not live
    navigation directions.
-   **Indoor attraction suggestions:** Uses dataset tags to support
    indoor and rainy-day recommendations.
-   **Evaluation and feedback:** Lets users rate itinerary relevance,
    add notes, and save evaluation records.
-   **Export:** Supports JSON itinerary export and CSV evaluation
    export.

## Technology Stack

  Technology   Purpose
  ------------ ----------------------------------------------
  Python       Application logic and data processing
  Streamlit    Web application interface
  Pandas       Reading, filtering, and processing CSV data
  Ollama       Running the language model locally
  Llama 3.2    Generating itinerary and assistant responses
  CSV          Storing attraction information
  JSON         Structured itinerary output and export

## Project Structure

``` text
AI-Powered-Personalized-Travel-Planner-Using-NLP/
├── app_v6.py
├── travel_planner_attractions_dataset.csv
├── requirements.txt
└── README.md
```

The filenames in your repository should match the filenames used by the
application. Add any other files your application imports or requires.

## Prerequisites

-   Python 3.10 or a compatible version
-   Ollama
-   Llama 3.2 model

## Installation and Setup

### 1. Clone the repository

``` bash
git clone https://github.com/deeksh20/AI-Powered-Personalized-Travel-Planner-Using-NLP.git
cd AI-Powered-Personalized-Travel-Planner-Using-NLP
```

### 2. Create and activate a virtual environment (recommended)

On Windows:

``` bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

If `requirements.txt` is present:

``` bash
pip install -r requirements.txt
```

If you do not have a requirements file yet, install the main packages
used by the application:

``` bash
pip install streamlit pandas ollama streamlit-mic-recorder
```

Install any additional packages imported by `app_v6.py` if needed.

### 4. Install and prepare Ollama

Install Ollama from <https://ollama.com/>.

Then run:

``` bash
ollama pull llama3.2
```

Check that the model is available:

``` bash
ollama list
```

Ollama must be running locally when using the AI features.

### 5. Run the application

From the project folder:

``` bash
streamlit run app_v6.py
```

Open the local URL displayed in the terminal.

## How to Use

1.  Enter or select a travel destination.
2.  Choose interests and trip duration.
3.  Enter the budget details.
4.  Select **Generate My Itinerary**.
5.  Review the day-wise itinerary and estimated budget.
6.  Ask travel-related questions using the AI Travel Assistant.
7.  Submit a relevance rating and optional feedback in the evaluation
    section.
8.  Download the itinerary as JSON or evaluation records as CSV, where
    available.

## Dataset

The application uses `travel_planner_attractions_dataset.csv`. Its
columns include:

-   `attraction_id`
-   `destination`
-   `attraction_name`
-   `category`
-   `keywords`
-   `estimated_entry_fee_inr_per_person`
-   `estimated_visit_hours`
-   `tags`
-   `source_url`
-   `cost_data_note`
-   `source_name`

Refer to the CSV file in the repository for the actual attraction
records.

## Important Notes and Limitations

-   Llama 3.2 runs locally through Ollama. A working Ollama installation
    and downloaded model are required for AI-powered features.
-   Entry fees, food, transport, accommodation, route distances, and
    travel times are estimates and may differ from actual prices and
    conditions.
-   Route distance and travel time are approximate calculations, not
    live map or navigation services.
-   Recommendations depend on the dataset content and the model's
    generated response.
-   The evaluation feature records user feedback. It does not
    automatically retrain or fine-tune Llama 3.2.

## Future Enhancements

-   Integrate live maps and real-time route information.
-   Retrieve current attraction timings, ticket prices, and availability
    from verified sources.
-   Expand the dataset to cover more destinations.
-   Add more comprehensive recommendation evaluation metrics.
-   Improve itinerary optimization using travel time, opening hours, and
    user preferences.

## Author

**Deekshitha K S**

## License

No license has been specified for this project. Contact the author
before reusing or redistributing the code or dataset.
