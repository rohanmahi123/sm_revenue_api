"""
SQLAlchemy models shared with Part 1 (processing pipeline).
These mirror the tables already in Supabase — read-only from this service's
perspective (auth only). Do NOT run migrations from here.
"""

from sqlalchemy import Boolean, Column, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Company(Base):
    __tablename__ = "companies"

    id                  = Column(Integer, primary_key=True, index=True)
    name                = Column(String(256), nullable=False)
    industry            = Column(String(128), nullable=True)
    slug                = Column(String(256), nullable=True)
    base_currency       = Column(String(10), nullable=True)
    reporting_currency  = Column(String(10), nullable=True)
    is_deleted          = Column(Boolean, default=False)


class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    company_id    = Column(Integer, nullable=False, index=True)
    name          = Column(String(256), nullable=True)
    email         = Column(String(256), nullable=False, index=True)
    password_hash = Column(String(512), nullable=False)
    role          = Column(String(64), default="analyst")
    is_active     = Column(Boolean, default=True)
    is_deleted    = Column(Boolean, default=False)
