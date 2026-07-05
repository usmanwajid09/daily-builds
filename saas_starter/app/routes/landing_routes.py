"""Marketing landing page (Milestone 2).

This starter is backend-first (a JSON API, not a rendered frontend app),
so the landing page is intentionally a single server-rendered page: pricing
copy pulled straight from the same PLANS catalogue the API enforces (so
the page can never drift out of sync with what the API actually allows),
plus a minimal vanilla-JS signup form that calls the real /api/auth/signup
endpoint -- a working demo, not just a mockup.
"""
from flask import Blueprint, current_app, render_template

from ..billing import PLANS

bp = Blueprint("landing", __name__)


@bp.get("/")
def landing():
    return render_template("landing.html", plans=PLANS)
