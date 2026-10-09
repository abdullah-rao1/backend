from __future__ import annotations

from sqlalchemy.orm import Session

from core.security import hash_password, verify_password
from models.orm.user import User


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: str) -> User | None:
    return db.get(User, user_id)


def create_user(db: Session, email: str, password: str, full_name: str | None = None) -> User:
    if get_user_by_email(db, email):
        raise ValueError("An account with this email already exists.")

    user = User(email=email, hashed_password=hash_password(password), full_name=full_name)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    user = get_user_by_email(db, email)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


def update_password(db: Session, user: User, new_password: str) -> None:
    user.hashed_password = hash_password(new_password)
    db.commit()


def delete_user(db: Session, user: User) -> None:
    db.delete(user)
    db.commit()
