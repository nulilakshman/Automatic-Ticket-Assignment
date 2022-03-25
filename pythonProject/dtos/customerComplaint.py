from typing import Optional

from pydantic import BaseModel


class CustomerComplaint(BaseModel):
    description: str
    predictedcategory: Optional[str]
