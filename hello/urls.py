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
    path("ask_tns", hello.views.ask_tns, name="ask_tns"),
    path("ask_basb", hello.views.ask_basb, name="ask_basb"),
    path("ask_drbb", hello.views.ask_drbb, name="ask_drbb"),
    path("question/<int:id>", hello.views.question, name="question"),
    path("db", hello.views.db, name="db"),
    path("admin/", admin.site.urls),
]
