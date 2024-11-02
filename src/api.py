from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from fastapi.responses import ORJSONResponse

from src.functions import (
    format_url,
    parse_response,
    predict_category,
    read_pickle,
    scrape_url,
)

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:3000/"
]

app.add_middleware(CORSMiddleware, allow_origins=origins)

handler = Mangum(app)

words_frequency = read_pickle("frequency_models/model.pickle")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/healthcheck/")
def healthy_condition() -> dict[str, str]:
    return {"status": "online"}


@app.post("/cat/")
async def predict_url(url: str) -> ORJSONResponse:
    url = format_url(url)
    response = [0, scrape_url(url, prediction=True)]
    html_content = parse_response(response)
    results = predict_category(words_frequency, html_content)
    results["response"] = response[1]  # type: ignore
    results["tokens"] = html_content[1]  # type: ignore
    return ORJSONResponse(content=results)
