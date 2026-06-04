from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session
from database import SessionLocal
from models import User, Company
from security import hash_password, verify_password
from jose import jwt, JWTError, ExpiredSignatureError
from datetime import datetime, timedelta
import os
import uuid

router = APIRouter()

# =========================
# CONFIG
# =========================
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 8

# =========================
# VALID CURRENCIES (MVP)
# =========================
VALID_CURRENCIES = {"USD", "AUD", "INR", "EUR", "GBP"}


# =========================
# DB Dependency
# =========================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =========================
# REQUEST MODELS
# =========================
class SignupRequest(BaseModel):
    company_name: str
    industry: str | None = None
    base_currency: str
    reporting_currency: str
    name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# =========================
# UTILS
# =========================
def generate_slug(name: str) -> str:
    base = name.lower().strip().replace(" ", "-")
    return f"{base}-{uuid.uuid4().hex[:4]}"


def validate_currency(currency: str) -> str:
    currency = currency.upper()
    if currency not in VALID_CURRENCIES:
        raise HTTPException(status_code=400, detail=f"Invalid currency: {currency}")
    return currency


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow()
    })

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# =========================
# SIGNUP
# =========================
@router.post("/signup")
def signup(data: SignupRequest, db: Session = Depends(get_db)):

    email = data.email.lower().strip()
    slug = generate_slug(data.company_name)

    base_currency = validate_currency(data.base_currency)
    reporting_currency = validate_currency(data.reporting_currency)

    try:
        # 🚨 Enforce GLOBAL email uniqueness
        existing_global_user = db.query(User).filter(
            User.email == email,
            User.is_deleted == False
        ).first()

        if existing_global_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        # Check if company exists
        company = db.query(Company).filter(
            Company.name == data.company_name,
            Company.is_deleted == False
        ).first()

        is_new_company = False

        if not company:
            company = Company(
                name=data.company_name,
                industry=data.industry,
                slug=slug,
                base_currency=base_currency,
                reporting_currency=reporting_currency
            )
            db.add(company)
            db.commit()
            db.refresh(company)
            is_new_company = True

        # Create user
        new_user = User(
            company_id=company.id,
            name=data.name.strip().title(),
            email=email,
            password_hash=hash_password(data.password),
            role="admin" if is_new_company else "analyst"
        )

        db.add(new_user)
        db.commit()

        return {
            "message": "Signup successful",
            "company_slug": company.slug,
            "base_currency": company.base_currency,
            "reporting_currency": company.reporting_currency
        }

    except Exception as e:
        db.rollback()
        raise e


# =========================
# LOGIN (EMAIL + PASSWORD ONLY)
# =========================
@router.post("/login")
def login(data: LoginRequest, db: Session = Depends(get_db)):

    email = data.email.lower().strip()

    user = (
        db.query(User)
        .filter(
            User.email == email,
            User.is_deleted == False,
            User.is_active == True
        )
        .first()
    )

    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({
        "sub": str(user.id),
        "user_id": user.id,
        "company_id": user.company_id,
        "role": user.role
    })

    return {
        "access_token": token,
        "token_type": "bearer"
    }


# =========================
# CURRENT USER
# =========================
def get_current_user(
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    try:
        token = authorization.replace("Bearer ", "").replace("bearer ", "")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        user_id = payload.get("user_id")
        company_id = payload.get("company_id")

        user = db.query(User).filter(
            User.id == user_id,
            User.company_id == company_id,
            User.is_deleted == False,
            User.is_active == True
        ).first()

        if not user:
            raise HTTPException(status_code=401, detail="User not found")

        return user

    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# =========================
# ROLE-BASED ACCESS
# =========================
def require_admin(user: User = Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# =========================
# LOGOUT (MVP)
# =========================
@router.post("/logout")
def logout():
    return {
        "message": "Logout successful. Delete token on client side."
    }