 storage_start = time.time()
        insert_sql = """
            INSERT INTO codecatalyst.embeddings (pdf_id, project_code, page_number, chunk_text, embedding)
            VALUES (%s, %s, %s, %s, %s)
        """
        for metadata, embedding in zip(chunk_metadata, embeddings):
            cursor.execute(
                insert_sql,
                (pdf_id, project_code, metadata['page_number'], metadata['text'], embedding)
            )
        conn.commit()
