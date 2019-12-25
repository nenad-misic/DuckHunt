
$(document).ready(function(){
    $('.stream-container').click(function() {
        $.ajax({
            url: "http://192.168.0.15:5000/shoot",
            type: 'GET',
            success: function(res) {
                if(res === 'bird'){
                    var shooter = $(".shooter")[0];
                    shooter.pause();
                    shooter.currentTime = 0;
                    shooter.play();
                }
            }
        });
    });
});