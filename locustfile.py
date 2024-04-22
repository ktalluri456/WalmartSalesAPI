from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def predict_sales(self):
        self.client.post("/predict", json={"date": "2022-01-01"})

    @task(3)
    def view_root(self):
        self.client.get("/")