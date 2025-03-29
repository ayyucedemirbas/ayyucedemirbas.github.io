// ekran hareket hızı
let move_speed = 3;
    
//yer çekimi
let gravity = 0.5;
    
let box = document.querySelector('.bird');
var fly = new Audio();
fly.src = "./jump.wav"   

var gameover = new Audio();
gameover.src = "./gameover.wav" 
var well_done = new Audio();
well_done.src = "./well_done.wav" 
let box_props = box.getBoundingClientRect();
let background =
    document.querySelector('.background')
            .getBoundingClientRect();
    
let score_val =
    document.querySelector('.score_val');
let message =
    document.querySelector('.message');
let score_title =
    document.querySelector('.score_title');
    

let game_state = 'Start';
    


document.addEventListener('keydown', (e) => {
    
  // Enter tusuna basıldığında oyunu başlat
  if (e.key == 'Enter' &&
      game_state != 'Play') {
    document.querySelectorAll('.pipe')
              .forEach((e) => {
      e.remove();
    });
    box.style.top = '40vh';
    game_state = 'Play';
    message.innerHTML = '';
    score_title.innerHTML = 'Score : ';
    score_val.innerHTML = '0';
    play();
  }
});
function play() {
  function move() {
      
  
    if (game_state != 'Play') return;
      
    // Getting reference to all the pipe elements
    let pipe_sprite = document.querySelectorAll('.pipe');
    pipe_sprite.forEach((element) => {
        
      let pipe_sprite_props = element.getBoundingClientRect();
      box_props = box.getBoundingClientRect();
        
      //borular ekrandan çıktığında sil
      if (pipe_sprite_props.right <= 0) {
        element.remove();
      } else {
        // kuş ve borular için çarpışma algılaması
        if (
          box_props.left < pipe_sprite_props.left +
          pipe_sprite_props.width &&
          box_props.left +
          box_props.width > pipe_sprite_props.left &&
          box_props.top < pipe_sprite_props.top +
          pipe_sprite_props.height &&
          box_props.top +
          box_props.height > pipe_sprite_props.top
        ) {
            
          //çarpışma durumunda oyunu bitir
          game_state = 'End';
          gameover.play();
          message.innerHTML = 'Press Enter.';
          message.style.left = '28vw';
          
          return;
        } else {
          //skoru artır
          if (
            pipe_sprite_props.right < box_props.left &&
            pipe_sprite_props.right + 
            move_speed >= box_props.left &&
            element.increase_score == '1'
          ) {
            score_val.innerHTML = +score_val.innerHTML + 1;
          }
         


           //mod alınca olmuyor sebebini çözemedim
            if(parseInt(score_val.innerHTML) == 10 ||
            parseInt(score_val.innerHTML) == 20
            ||
            parseInt(score_val.innerHTML) == 30
            ||
            parseInt(score_val.innerHTML) == 40
            ||
            parseInt(score_val.innerHTML) == 50
            ||
            parseInt(score_val.innerHTML) == 60||
            parseInt(score_val.innerHTML) == 70||
            parseInt(score_val.innerHTML) == 80||
            parseInt(score_val.innerHTML) == 90||
            parseInt(score_val.innerHTML) == 100||
            parseInt(score_val.innerHTML) == 110||
            parseInt(score_val.innerHTML) == 120||
            parseInt(score_val.innerHTML) == 130||
            parseInt(score_val.innerHTML) == 140||
            parseInt(score_val.innerHTML) == 150||
            parseInt(score_val.innerHTML) == 160||
            parseInt(score_val.innerHTML) == 170||
            parseInt(score_val.innerHTML) == 180||
            parseInt(score_val.innerHTML) == 190||
            parseInt(score_val.innerHTML) == 200 ){
            //let message = document.querySelector('.message'); //oyun duruyo
            message.innerHTML = 'Well done';
            message.style.left = '28vw';
            well_done.play();
            //move_speed+=0.1;
            //message.remove();

            document.addEventListener('keydown', (e) => {
              if (e.key == 'ArrowUp' || e.key == ' ') {
                message.remove();
              }
            });

            

            
            }


            element.style.left = 
            pipe_sprite_props.left - move_speed + 'px';


        }
      }
    });
  
    requestAnimationFrame(move);
  }
  requestAnimationFrame(move);
  let message =
    document.querySelector('.message');
  let box_dy = 0;
  function apply_gravity() {
    if (game_state != 'Play') return;
    box_dy = box_dy + gravity;
    document.addEventListener('keydown', (e) => {
      if (e.key == 'ArrowUp' || e.key == ' ') {
        box_dy = -7.6;
        fly.play();
      }
    });
  
    //kuş ve üst alt ekran arası çarpışma algılaması
  
    if (box_props.top <= 0 || box_props.bottom >= background.bottom) {
        game_state = 'End';
      message.innerHTML = 'Press enter';
      message.style.left = '28vw';
      return;
    }
    box.style.top = box_props.top + box_dy + 'px';
    box_props = box.getBoundingClientRect();
    requestAnimationFrame(apply_gravity);
  }
  requestAnimationFrame(apply_gravity);
  
  let pipe_seperation = 0;
    
  // üst-alt borular arası boşluk
  let pipe_gap = 35;
  function create_pipe() {
    if (game_state != 'Play') return;
      
    // Create another set of pipes
    // if distance between two pipe has exceeded
    // a predefined value
    if (pipe_seperation > 150) {
      pipe_seperation = 0
        
      // boruları oluşturmak için y ekseninde rastlege konumlar belirle
      let pipe_pos = Math.floor(Math.random() * 43) + 6;
      let pipe_sprite_inv = document.createElement('div');
      pipe_sprite_inv.className = 'pipe';
      pipe_sprite_inv.style.top = pipe_pos - 68 + 'vh';
      pipe_sprite_inv.style.left = '100vw';
        
      
      document.body.appendChild(pipe_sprite_inv);
      let pipe_sprite = document.createElement('div');
      pipe_sprite.className = 'pipe';
      pipe_sprite.style.top = pipe_pos + pipe_gap + 'vh';
      pipe_sprite.style.left = '100vw';
      pipe_sprite.increase_score = '1';
        
      
      document.body.appendChild(pipe_sprite);
    }
    pipe_seperation++;
    requestAnimationFrame(create_pipe);
  }
  requestAnimationFrame(create_pipe);
}