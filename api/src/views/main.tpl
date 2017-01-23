<!doctype html>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.1.1/jquery.min.js"></script>

<form>
    <textarea rows="20" cols="100" name="text" id="textarea"></textarea>
    <br/>
    <input type="button" value="Classify" id="submit-button"/>
</form>
<br/>
<b>Classification Category:</b>
<div id="classification_category">

</div>
<b>Response Time (sec)</b>
<div id="response_seconds">

</div>

<script>
    $('#submit-button').click(function(){

        $.ajax({
            url:'api/classify',
            type:"POST",
            data: JSON.stringify({text: $('#textarea').val()}),
            contentType:"application/json; charset=utf-8",
            dataType:"json",
            success: function(data){
                $('#classification_category').text(data.class);
                $('#response_seconds').text(data.response_sec);
            }
        });

    })
</script>