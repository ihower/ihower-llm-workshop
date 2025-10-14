from agents import SQLiteSession
from agents.items import TResponseInputItem

class CustomSQLiteSession(SQLiteSession):

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:

        # Call parent's get_items to get all items
        items = await super().get_items(limit=limit)

        # TBD

        return items
