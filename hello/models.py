from django.db import models

class Question(models.Model):
    question = models.CharField(max_length=140)
    context = models.TextField(null=True, blank=True)
    answer = models.TextField(max_length=1000, null=True, blank=True)
    created_at = models.DateTimeField("date created", auto_now_add=True)
    ask_count = models.IntegerField(default=1)
