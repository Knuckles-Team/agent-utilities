import ladybug

try:
    db1 = ladybug.Database("test_db")
    print("DB1 opened write")
    try:
        db2 = ladybug.Database("test_db")
        print("DB2 opened write")
    except Exception as e:
        print("DB2 write failed:", e)
        try:
            db3 = ladybug.Database("test_db", read_only=True)
            print("DB3 opened read-only")
        except Exception as e:
            print("DB3 read-only failed:", e)
except Exception as e:
    print("DB1 failed:", e)
