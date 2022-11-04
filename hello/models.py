from django.db import models

class Question(models.Model):
    question = models.CharField(max_length=140)
    answer = models.TextField(max_length=1000, null=True, blank=True)
    created_at = models.DateTimeField("date created", auto_now_add=True)
    ask_count = models.IntegerField(default=1)
    audio_src_url = models.CharField(default="", max_length=255, null=True, blank=True)
