import hello
def test_hello():
    try:
        assert hello.hello_world()==1
    except:
        print ("An exception was thrown!")
        assert False, f"raised an exception"
