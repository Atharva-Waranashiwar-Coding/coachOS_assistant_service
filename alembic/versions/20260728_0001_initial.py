"""create assistant storage"""

from alembic import op

revision = "20260728_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("""
      CREATE TABLE assistant_conversations (
        id uuid PRIMARY KEY, organization_id uuid NOT NULL, coach_id uuid NOT NULL,
        title varchar(200) NOT NULL, summary text NOT NULL,
        archived_at timestamptz, created_at timestamptz DEFAULT now(), updated_at timestamptz DEFAULT now()
      )
    """)
    op.execute("""
      CREATE TABLE assistant_messages (
        id uuid PRIMARY KEY, conversation_id uuid NOT NULL REFERENCES assistant_conversations(id) ON DELETE CASCADE,
        role varchar(16) NOT NULL, content text NOT NULL, citations jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz DEFAULT now()
      )
    """)
    op.execute("""
      CREATE TABLE assistant_chunks (
        id uuid PRIMARY KEY, stable_key varchar(300) NOT NULL UNIQUE, chunk_type varchar(64) NOT NULL,
        content text NOT NULL, embedding vector(128) NOT NULL, metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        organization_id uuid NOT NULL, coach_id uuid NOT NULL, athlete_id uuid, entity_id varchar(100) NOT NULL,
        created_at timestamptz DEFAULT now(), updated_at timestamptz DEFAULT now()
      )
    """)
    op.execute("CREATE INDEX ix_assistant_conversations_organization_id ON assistant_conversations (organization_id)")
    op.execute("CREATE INDEX ix_assistant_conversations_coach_id ON assistant_conversations (coach_id)")
    op.execute("CREATE INDEX ix_assistant_messages_conversation_id ON assistant_messages (conversation_id)")
    op.execute("CREATE INDEX ix_assistant_chunks_chunk_type ON assistant_chunks (chunk_type)")
    op.execute("CREATE INDEX ix_assistant_chunks_organization_id ON assistant_chunks (organization_id)")
    op.execute("CREATE INDEX ix_assistant_chunks_coach_id ON assistant_chunks (coach_id)")
    op.execute("CREATE INDEX ix_assistant_chunks_athlete_id ON assistant_chunks (athlete_id)")
    op.execute("CREATE INDEX ix_assistant_chunks_entity_id ON assistant_chunks (entity_id)")


def downgrade() -> None:
    op.execute("DROP TABLE assistant_chunks")
    op.execute("DROP TABLE assistant_messages")
    op.execute("DROP TABLE assistant_conversations")
