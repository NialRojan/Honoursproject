from fastapi import FastAPI

app = FastAPI(title="Dark Pattern Detection Engine")


@app.get("/health")
def health():
    return {"status": "ok"}
