from django.urls import path, include

from django.contrib import admin

admin.autodiscover()

import hello.views

# To add a new path, first import the app:
# import blog
#
# Then add the new path:
# path('blog/', blog.urls, name="blog")
#
# Learn more here: https://docs.djangoproject.com/en/2.1/topics/http/urls/

urlpatterns = [
    path("", hello.views.index, name="index"),
    path("ask", hello.views.ask, name="ask"),
    path("question/<int:id>", hello.views.question, name="question"),
    path("db/", hello.views.db, name="db"),
    path("real_answer", hello.views.real_answer, name="real_answer"),
    path("delete_question", hello.views.delete_question, name="delete_question"),
    path("metadata.jsonl", hello.views.metadata, name="metadata"),
    path("admin/", admin.site.urls),
]
