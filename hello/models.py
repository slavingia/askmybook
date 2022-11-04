from django.db import models

class Question(models.Model):
    question = models.CharField(max_length=140)
    answer = models.CharField(max_length=280)
    created_at = models.DateTimeField("date created", auto_now_add=True)

class Greeting(models.Model):
    when = models.DateTimeField("date created", auto_now_add=True)
